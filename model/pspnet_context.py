from dis import dis
from numpy import logical_and
import torch
from torch import nn
import torch.nn.functional as F

from collections import OrderedDict

import sys, os

sys.path.insert(0, os.path.abspath('.'))
from model.pspnet import PSPNet
from util.classification_utils import extract_mask_distributions


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


class DistributionMatch(nn.Module):
    """
    Given a distribution of how pixels should be assigned to each class in the image
    Correct the distribution of logits such that the distributions are equal
    """
    def __init__(self, correction_mode = "softmax", top_k = 150, dist_dim = "all", norm="bn"):
        """
        correction_mode: one of 'softmax' or 'logits', indicating which should be set equal to the class pixel distribution
        top_k: how many classes in target distribution
        dist_dim: one of 'all' or 'k', whether or not the prediction head is a softmax over num_classes or k
        norm: one of 'bn', 'ln' or None, indicating if BatchNorm/LayerNorm should be used for penultimate layers
        """
        super(DistributionMatch, self).__init__()
        assert correction_mode in ["logits", "softmax"]
        assert dist_dim in ["all", "k"]
        assert norm in ["bn", "ln", "none"]
        self.dist_dim = dist_dim
        self.top_k = top_k
        self.correction_mode = correction_mode

        self.norm = (norm != "none")

        self.layer1 = nn.Conv2d(512, 150, kernel_size=1, bias=True)


    def forward(self, pre_cls, logits, distribution, label=None):
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
            # learn dist based on pre_cls features and existing logit prediction
            x = self.layer1(pre_cls) + logits
            x = nn.Softmax(dim=1)(x)
            # get predicted distribution
            distribution = nn.AdaptiveAvgPool2d((1,1))(x)
            # convert to top-k distribution
            distribution = distribution * top_k_mask
            distribution = distribution / distribution.sum(dim=1, keepdims=True)
            k_distribution, _ = torch.topk(distribution, dim=1, k=self.top_k, largest=True, sorted=True)

            # compute loss
            label = label * top_k_mask  # GT distribution of top_k predicted classes
            label = label / (label.sum(dim=1, keepdims=True) + 1e-12)  # renormalize so that relative distribution among k sums to 1
            k_label, _ = torch.topk(label, dim=1, k=self.top_k, largest=True, sorted=True)

            loss = categorical_cross_entropy(k_distribution, k_label)
            # if self.dist_dim == "all":
            #     loss = categorical_cross_entropy(distribution, label)  # CCE across vector dim C
            # else:
            #     loss = earth_mover_distance(k_distribution, k_label)  # EMD across vector dim K

        dist_residual = distribution - softmax_distribution
        dist_residual = dist_residual * top_k_mask  # mask out updates for classes outside top k
        corrected_distribution = softmax + dist_residual # if self.correction_mode == "softmax" else logits + dist_residual_upsample

        return corrected_distribution, loss

class DropoutEnsemble(nn.Module):
    def __init__(self, cls, n_heads=10):
        super(DropoutEnsemble, self).__init__()
        self.cls = cls
        self.n_heads = n_heads

    def forward(self, x):
        # base forward pass
        self.cls.eval()
        logits = self.cls(x)
        softmax = nn.Softmax(dim=1)(logits)
        softmax_distribution = nn.AdaptiveAvgPool2d((1, 1))(softmax)

        # dropout ensemble
        dropouts = []
        self.cls[3].train()  # enable dropout layer
        for _ in range(self.n_heads):
            x_tmp = self.cls(x)
            x_tmp = nn.Softmax(dim=1)(x_tmp)
            x_tmp = nn.AdaptiveAvgPool2d((1, 1))(x_tmp)
            dropouts.append(x_tmp)
        dropouts = torch.stack(dropouts, dim=0)
        dropout_dist = torch.mean(dropouts, dim=0)

        dist_correction = dropout_dist - softmax_distribution
        dist_correction = F.interpolate(dist_correction, size=softmax.size()[2:], mode="nearest")

        corrected_sm = softmax + dist_correction

        return corrected_sm

class PSPNetContext(nn.Module):
    """
    transfer learn pspnet with trained classification heads to benchmark classification head + presoftmax methods
    """
    def __init__(self, layers=50, classes=150, zoom_factor=8, pspnet_weights=None, top_k=150, dist_dim="all", norm="none"):
        super(PSPNetContext, self).__init__()
        self.pspnet = PSPNet(layers=layers, classes=classes, zoom_factor=zoom_factor, pretrained=False)
        if pspnet_weights is not None:
            checkpoint = torch.load(pspnet_weights)['state_dict']
            fix_state_dict = OrderedDict()
            for k, v in checkpoint.items():
                name = k[7:] # remove 'module.' of dataparallel
                fix_state_dict[name] = v
            self.pspnet.load_state_dict(fix_state_dict, strict=True)
        for param in self.pspnet.parameters():
            param.requires_grad = False
        self.top_k = top_k
        self.dist_dim = dist_dim
        self.combo = DistributionMatch(top_k=top_k, dist_dim=dist_dim, norm=norm) # ContextLogitCombination()
        self.pred = DropoutEnsemble(self.pspnet.cls, n_heads=10)
        self.loss = earth_mover_distance
        self.zoom_factor = zoom_factor
        self.classes = classes

    def forward(self, x, y=None, distributions=None):
        """
        y[0] = segmentation label
        y[1] = distribution label
        distributions = GT distributions for inference
        """
        segmentation_label = y[0] if y is not None else y
        distribution_label = y[1] if y is not None else y

        self.pspnet.eval() # inference mode for BN/Dropout in base network
        x_size = x.size()
        assert (x_size[2]-1) % 8 == 0 and (x_size[3]-1) % 8 == 0
        h = int((x_size[2] - 1) / 8 * self.zoom_factor + 1)
        w = int((x_size[3] - 1) / 8 * self.zoom_factor + 1)

        x = self.pspnet.layer0(x)
        x = self.pspnet.layer1(x)
        x = self.pspnet.layer2(x)
        x = self.pspnet.layer3(x)
        x = self.pspnet.layer4(x)

        x_tmp = self.pspnet.ppm(x)

        # x_alt = self.pred(x_tmp) # dropout exp.

        for i in range(len(self.pspnet.cls)-1):  # up to segmentation-level conv
            x_tmp = self.pspnet.cls[i](x_tmp)  
        
        x = self.pspnet.cls[-1](x_tmp)  # segmentation logits
        x_alt, dist_loss = self.combo(pre_cls=x_tmp, logits=x, distribution=distributions, label=distribution_label)        

        if self.zoom_factor != 1:
            x = F.interpolate(x, size=(h, w), mode='bilinear', align_corners=True)
            x_alt = F.interpolate(x_alt, size=(h, w), mode='bilinear', align_corners=True)

        if self.training:
            # main_loss = self.pspnet.pspnet.criterion(x_alt, segmentation_label)
            total_loss = dist_loss # + main_loss (end-to-end loss)
            return x_alt.max(1)[1], dist_loss
        else:
            return x, x_alt, dist_loss
