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

class DistributionMatch(nn.Module):
    """
    Given a distribution of how pixels should be assigned to each class in the image
    Correct the distribution of logits such that the distributions are equal
    """
    def __init__(self, correction_mode = "softmax", top_k = 150):
        super(DistributionMatch, self).__init__()
        assert correction_mode in ["logits", "softmax"]
        self.top_k = top_k
        self.correction_mode = correction_mode
        self.layer1 = nn.Conv2d(512, 150, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(150)
        self.layer2 = nn.Conv2d(150, 150, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(150)
        self.layer3 = nn.Conv2d(150, 150, kernel_size=1)
        

    def forward(self, pre_cls, logits, distribution, ref=None):
        """
        pre_cls - 512 channel pre-cls feature map
        logits - NUM_CLASS channel logits
        distribution - image-level class distribution (NxCx1x1) tensor
        """
        # with ground truth or learned distribution
        softmax = nn.Softmax(dim=1)(logits)
        softmax_distribution = nn.AdaptiveAvgPool2d((1, 1))(softmax)
        top_k = torch.topk(softmax_distribution, dim=1, k=self.top_k, largest=True, sorted=True)[0]
        top_k_threshold = torch.min(top_k, dim=1, keepdim=True)[0]
        if distribution is None:
            # learn distribution based on features
            x1 = self.layer1(pre_cls)
            x1 = self.bn1(x1)
            x1 = nn.ReLU()(x1)
            x2 = self.layer2(logits)
            x2 = self.bn2(logits)
            x2 = nn.ReLU()(x2)
            x3 = nn.AdaptiveAvgPool2d((1, 1))(x1 + x2)
            x3 = self.layer3(x3)
            offsets = nn.Sigmoid()(x3)
            distribution = softmax_distribution * offsets
            distribution = torch.where(softmax_distribution > top_k_threshold, distribution, torch.zeros_like(distribution))

        positive_logits = nn.ReLU()(logits)
        positive_logits = nn.AdaptiveAvgPool2d((1, 1))(positive_logits)
        positive_logits_distribution = positive_logits / positive_logits.sum(axis=1, keepdims=True)
        target_distribution = softmax_distribution if self.correction_mode == "softmax" else positive_logits_distribution
        dist_residual = distribution - target_distribution
        dist_residual = torch.where(distribution < 1e-6, torch.zeros_like(dist_residual), dist_residual) # don't apply updates when distribution label dne
        dist_residual_upsample = F.interpolate(dist_residual, size=softmax.size()[2:], mode="nearest")
        corrected_distribution = softmax + dist_residual_upsample # if self.correction_mode == "softmax" else logits + dist_residual_upsample

        return corrected_distribution, distribution

def categorical_cross_entropy(y_pred, y_true, weights=None, smooth=0):
    """
    Weighted/Smoothed CCE
    """
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

class PSPNetContext(nn.Module):
    """
    transfer learn pspnet with trained classification heads to benchmark classification head + presoftmax methods
    """
    def __init__(self, layers=50, classes=150, zoom_factor=8, pspnet_weights=None):
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
        self.combo = DistributionMatch(top_k=5) # ContextLogitCombination()
        self.zoom_factor = zoom_factor
        self.classes = classes

    def forward(self, x, y=None, distributions=None):
        """
        y[0] = segmentation label
        y[1] = distribution label
        """
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

        for i in range(len(self.pspnet.cls)-1):  # up to segmentation-level conv
            x_tmp = self.pspnet.cls[i](x_tmp)  
        
        x = self.pspnet.cls[-1](x_tmp)  # segmentation logits
        x_alt, dist_pred = self.combo(x_tmp, x, distributions)

        if self.zoom_factor != 1:
            x = F.interpolate(x, size=(h, w), mode='bilinear', align_corners=True)
            x_alt = F.interpolate(x_alt, size=(h, w), mode='bilinear', align_corners=True)

        if self.training:
            segmentation_label = y[0]
            distribution_label = y[1]
            loss = categorical_cross_entropy(dist_pred, distribution_label)
            # main_loss = self.pspnet.pspnet.criterion(x_alt, segmentation_label)
            return x_alt.max(1)[1], loss
        else:
            return x, x_alt