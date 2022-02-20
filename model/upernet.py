from distutils.command.config import config
from multiprocessing import pool
from tkinter.tix import Tree
from mmseg.apis import inference_segmentor, init_segmentor
import mmcv
import torch
from torch import nn
import torch.nn.functional as F
from mmseg.ops import resize

resnet_config = "exp/ade20k/upernet50/model/config.py"
resnet_checkpoint = "exp/ade20k/upernet50/model/upernet_r50.pth"
swin_config = "exp/ade20k/upernet_swin/model/config.py"
swin_checkpoint = "exp/ade20k/upernet_swin/model/upernet_swin_t.pth"

def categorical_cross_entropy(y_pred, y_true, weights=None, smooth=0):
    """CCE Loss w/Weighting+Smoothing Support"""
    y_pred = torch.clamp(y_pred, 1e-9, 1 - 1e-9)
    ce = None
    if smooth > 0:
        uniform = torch.ones_like(y_true) / 150
        true_mask = 1.0 * (y_true > 0)
        uniform *= true_mask
        uniform /= uniform.sum(axis=1, keepdims=True)
        y_true = y_true + (smooth * uniform)
        y_true = y_true / y_true.sum(axis=1, keepdims=True)
    if weights is not None:
        ce = -(weights * (y_true * torch.log(y_pred))).sum(dim=1)
    else: 
        ce = -(y_true * torch.log(y_pred)).sum(dim=1)
    return ce.mean()

def earth_mover_distance(y_pred, y_true):
    """EMD Loss"""
    cdf_pred = torch.cumsum(y_pred, dim=1)
    cdf_true = torch.cumsum(y_true, dim=1)
    # correct for when no label in image, 0-sum distribution
    valid_mask = torch.sum(y_true, dim=1, keepdim=True)  
    cdf_pred = cdf_pred * valid_mask
    return torch.mean(torch.square(cdf_pred - cdf_true))

class FiLM(nn.Module):
    def __init__(self, num_layers=1):
        """
        num_layers: number of layers for context embedding before FiLM combination
        """
        super(FiLM, self).__init__()
        self.num_layers = num_layers
        assert num_layers in [1, 2]
        self.layer1 = nn.Conv2d(150, 512*2, kernel_size=1, bias=True if num_layers == 1 else 2)
        self.norm1 = nn.LayerNorm([1024, 1, 1])
        self.layer2 = nn.Conv2d(512*2, 512*2, kernel_size=1, bias=True)

    
    def forward(self, pre_cls, context):
        """
        pre_cls - 512 channel feature map
        context - NUM_CLASS x 1 x 1 context vector (either classification or distribution)
        """
        context_embedding = self.layer1(context)
        if self.num_layers > 1:
            context_embedding = nn.ReLU()(context_embedding)
            context_embedding = self.layer2(context_embedding)
        film_scale, film_bias = torch.split(context_embedding, [512, 512], dim=1)
        film_features = (pre_cls * film_scale) + film_bias
        film_features = nn.ReLU()(film_features)
        return film_features

class ContextHead(nn.Module):
    def __init__(self, num_layers=1):
        """
        num_layers: number of layers for context embedding before FiLM combination
        """
        super(ContextHead, self).__init__()
        self.num_layers = num_layers
        assert num_layers in [1, 2]
        layer1_out = 150 if num_layers == 1 else 512
        self.layer1 = nn.Conv2d(512, layer1_out, kernel_size=1, bias=True if num_layers == 1 else 2)
        self.norm1 = nn.LayerNorm([512, 128, 128])
        self.layer2 = nn.Conv2d(layer1_out, 150, kernel_size=1, bias=True)

    def forward(self, pre_cls):
        """
        pre_cls - 512 channel feature map
        """
        x = pre_cls
        if self.num_layers == 1:
            x = nn.AdaptiveAvgPool2d((1,1))(x)
            x = self.layer1(x)
            x = nn.Sigmoid()(x)
            return x
        else:
            x = self.layer1(x)
            x = self.norm1(x)
            x = nn.ReLU()(x)
            x = nn.AdaptiveAvgPool2d((1,1))(x)
            x = self.layer2(x)
            x = nn.Sigmoid()(x)
            return x

class DistributionCorrection(nn.Module):
    def __init__(self, top_k=5):
        super(DistributionCorrection, self).__init__()
        self.top_k = top_k
    
    def forward(self, logits, distribution):
        """
        logits - NUM_CLASS channel logits
        distribution - image-level class distribution (NxCx1x1) tensor
        """
        # with ground truth or learned distribution
        softmax = nn.Softmax(dim=1)(logits)
        softmax_distribution = nn.AdaptiveAvgPool2d((1, 1))(softmax)
        top_k, ind = torch.topk(softmax_distribution, dim=1, k=self.top_k, largest=True, sorted=True)
        top_k_threshold = torch.min(top_k, dim=1, keepdim=True)[0]
        top_k_mask = torch.where(softmax_distribution > top_k_threshold, torch.ones_like(softmax_distribution), torch.zeros_like(softmax_distribution))
        
        k_softmax = softmax_distribution * top_k_mask
        k_softmax = k_softmax / k_softmax.sum(dim=1, keepdims=True)
        k_softmax, _ = torch.topk(k_softmax, dim=1, k=self.top_k, largest=True, sorted=True)
        
        dist_residual = distribution - softmax_distribution
        dist_residual = dist_residual * top_k_mask  # mask out updates for classes outside top k
        corrected_distribution = softmax + dist_residual

        k_softmax_alt = nn.AdaptiveAvgPool2d((1,1))(corrected_distribution) * top_k_mask
        k_softmax_alt = k_softmax_alt / k_softmax_alt.sum(dim=1, keepdims=True)
        k_softmax_alt, _ = torch.topk(k_softmax_alt, dim=1, k=self.top_k, largest=True, sorted=True)

        label = distribution * top_k_mask  # GT distribution of top_k predicted classes
        label = label / (label.sum(dim=1, keepdims=True) + 1e-12)  # renormalize so that relative distribution among k sums to 1
        k_label, _ = torch.topk(label, dim=1, k=self.top_k, largest=True, sorted=True)

        return corrected_distribution, k_label, k_softmax, k_softmax_alt

class DistributionMatch(nn.Module):
    """
    Given a distribution of how pixels should be assigned to each class in the image
    Correct the distribution of logits such that the distributions are equal
    """
    def __init__(self, top_k=5, init_weights=None):
        """
        correction_mode: one of 'softmax' or 'logits', indicating which should be set equal to the class pixel distribution
        top_k: how many classes in target distribution
        dist_dim: one of 'all' or 'k', whether or not the prediction head is a softmax over num_classes or k
        norm: one of 'bn', 'ln' or None, indicating if BatchNorm/LayerNorm should be used for penultimate layers
        """
        super(DistributionMatch, self).__init__()
        self.top_k = top_k

        self.layer1 = nn.Conv2d(512, 150, kernel_size=1, bias=True)
        if init_weights is not None:
            self.layer1.weight.data = init_weights


    def forward(self, pre_cls, logits, distribution=None, label=None):
        """
        pre_cls - 512 channel pre-cls feature map
        logits - NUM_CLASS channel logits
        distribution - image-level class distribution (NxCx1x1) tensor
        label - distribution label for computing loss
        """
        # with ground truth or learned distribution
        softmax = nn.Softmax(dim=1)(logits)
        softmax_distribution = nn.AdaptiveAvgPool2d((1, 1))(softmax)
        top_k, ind = torch.topk(softmax_distribution, dim=1, k=self.top_k, largest=True, sorted=True)
        top_k_threshold = torch.min(top_k, dim=1, keepdim=True)[0]
        top_k_mask = torch.where(softmax_distribution > top_k_threshold, torch.ones_like(softmax_distribution), torch.zeros_like(softmax_distribution))
        
        k_distribution = None
        loss = None

        if distribution is None:
            x = nn.AdaptiveAvgPool2d((1,1))(pre_cls)
            x = self.layer1(x)
            x = nn.Sigmoid()(x)

            x = logits * x
            distribution = nn.Softmax(dim=1)(logits)

            # convert to top-k distribution
            distribution = distribution * top_k_mask
            distribution = distribution / distribution.sum(dim=1, keepdims=True)
            k_distribution, _ = torch.topk(distribution, dim=1, k=self.top_k, largest=True, sorted=True)

            # compute loss
            label = label * top_k_mask  # GT distribution of top_k predicted classes
            label = label / (label.sum(dim=1, keepdims=True) + 1e-12)  # renormalize so that relative distribution among k sums to 1
            k_label, _ = torch.topk(label, dim=1, k=self.top_k, largest=True, sorted=True)

            loss = earth_mover_distance(k_distribution, k_label) # + categorical_cross_entropy(distribution, label)

        dist_residual = distribution - softmax_distribution
        dist_residual = dist_residual * top_k_mask  # mask out updates for classes outside top k
        corrected_distribution = softmax + dist_residual # if self.correction_mode == "softmax" else logits + dist_residual_upsample

        return corrected_distribution, loss

class UPerNet(nn.Module):
    def __init__(self, backbone="resnet", init_weights=None, gt_dist=False, film=False, learn_context=False, context_layers=1, film_layers=1, k=5):
        super(UPerNet, self).__init__()
        assert backbone in ["resnet", "swin"]
        config = resnet_config if backbone == "resnet" else swin_config
        checkpoint = resnet_checkpoint if backbone == "resnet" else swin_checkpoint
        self.model = init_segmentor(config, checkpoint, device='cuda:0')
        self.backbone = self.model.backbone
        self.decode_head = self.model.decode_head
        self.dist_head = DistributionMatch(top_k=k, init_weights=self.decode_head.conv_seg.weight.data)
        self.film_head = FiLM(num_layers=film_layers)
        self.context_head = ContextHead(num_layers=context_layers)
        self.learn_context=learn_context
        self.gt_dist = gt_dist
        self.film = film
        self.seg_loss = nn.CrossEntropyLoss(ignore_index=255)
        self.k = k

        if init_weights is not None:
            checkpoint = torch.load(init_weights)["state_dict"]
            self.load_state_dict(checkpoint, strict=False)

    def upernet_forward(self, inputs):
        inputs = self.decode_head._transform_inputs(inputs)

        # build laterals
        laterals = [
            lateral_conv(inputs[i])
            for i, lateral_conv in enumerate(self.decode_head.lateral_convs)
        ]

        laterals.append(self.decode_head.psp_forward(inputs))

        # build top-down path
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            prev_shape = laterals[i - 1].shape[2:]
            laterals[i - 1] = laterals[i - 1] + resize(
                laterals[i],
                size=prev_shape,
                mode='bilinear',
                align_corners=self.decode_head.align_corners)

        # build outputs
        fpn_outs = [
            self.decode_head.fpn_convs[i](laterals[i])
            for i in range(used_backbone_levels - 1)
        ]
        # append psp feature
        fpn_outs.append(laterals[-1])

        for i in range(used_backbone_levels - 1, 0, -1):
            fpn_outs[i] = resize(
                fpn_outs[i],
                size=fpn_outs[0].shape[2:],
                mode='bilinear',
                align_corners=self.decode_head.align_corners)
        fpn_outs = torch.cat(fpn_outs, dim=1)
        pre_cls = self.decode_head.fpn_bottleneck(fpn_outs)
        logits = self.decode_head.cls_seg(pre_cls)
        return pre_cls, logits

    def forward(self, x, y=None, context=None):
        self.backbone.eval()
        self.decode_head.eval()
        loss = 0
        h, w = x.size()[2:] 
        x = self.backbone(x)
        pre_cls, x = self.upernet_forward(x) # pre logit features and logits

        # LEARNED CONTEXT
        if self.learn_context:
            context_pred = self.context_head(pre_cls)
            context_loss = categorical_cross_entropy(context_pred, context)

            # return if only training for classification
            if not self.film:
                return context_pred, context_loss

            # otherwise use learned context for film
            else:
                context = context_pred

        # LEARNED FILM ADJUSTMENT WITH DISTRIBUTION / CLASSIFICATION
        if self.film and context is not None:
            pre_cls = self.film_head(pre_cls, context)
            x = self.decode_head.cls_seg(pre_cls) 
            x = F.interpolate(x, size=(h,w), mode="bilinear", align_corners=self.decode_head.align_corners)
            loss = loss + self.seg_loss(x, y)
            return x, loss

        # PARAMETER FREE CORRECTION WITH GT_DISTRIBUTION
        if self.gt_dist and context is not None:
            x, k_label, k_softmax, k_softmax_alt = DistributionCorrection(self.k)(x, context)
            return x, k_label, k_softmax, k_softmax_alt
        
        return x, loss   


if __name__ == "__main__":
    x = torch.rand(size=(8, 3, 512, 512)).cuda()
    model = UPerNet(backbone="swin", init_weights="upernet_swin_classification_9_v2.pth")
    pred = model.forward(x)
    print("hello")