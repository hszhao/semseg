from dis import dis
from numpy import logical_and
import torch
from torch import nn
import torch.nn.functional as F

from collections import OrderedDict

import sys, os
sys.path.insert(0, os.path.abspath('.'))
from model.pspnet_c import PSPNetContext
from util.classification_utils import extract_mask_distributions

class ContextLogitCombination(nn.Module):
    def __init__(self, classes=150, agg_dim=(6,6)):
        super(ContextLogitCombination, self).__init__()
        self.classes = classes
        self.agg_dim = agg_dim

    def forward(self, features, logits, out_size):
        # upsample all scales to agg dim (6x6)
        upsampled = []
        for scale in logits:
            # change classifications to logits
            upsampled.append(F.interpolate(scale, size=self.agg_dim, mode='nearest'))
        stack = torch.stack(upsampled)
        # upsampled + linearly combined classifications 
        # classification_logits = torch.mean(stack, dim=0)  # take mean, could be a learned weighted (1 weights for each scale) sum also
        classification_logits = stack[-1]
        classification_logits = F.normalize(classification_logits, p=1, dim=1)
        classification_logits = F.interpolate(classification_logits, size=out_size, mode="bilinear", align_corners=True)

        # l1 norm single scale
        # Val result: mIoU/mAcc/allAcc 0.4242/0.5334/0.7971
        # Val result2: mIoU/mAcc/allAcc 0.3968/0.4988/0.7872
        classification_logits = classification_logits

        segmentation_logits = features
        segmentation_logits = F.normalize(features, p=1, dim=1)

        return classification_logits + segmentation_logits


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
        self.layer1 = nn.Conv2d(150, 150, kernel_size=1)
        self.layer2 = nn.Conv2d(150, top_k-1, kernel_size=1)
        

    def forward(self, features, distribution):
        """
        this iteration assumes only image-wide distribution statistics are learned / passed in
        e.g. distribution is an NxCx1x1 tensor
        """
        if distribution is None:
            # learn distribution based on features
            # top_k class prediction is linear projection of existing pre-softmax features
            x = nn.AdaptiveAvgPool2d((1, 1))(features)
            x = self.layer1(x) # does this need to be here? 
            distribution = nn.Softmax(dim=1)(x)
            # top_k_ind = nn.topk(x, self.top_k, dim=1)[1]
            # proportions = self.layer2(x)
            # distribution = torch.ones(size=x.shape)
            # left_over = torch.ones_like(distribution)
            # for i in range(self.k):
            #     predicted_proportion = proportions[:, i, :, :]
            #     distribution_prop = predicted_proportion * left_over
            #     distribution[:,top_k_ind[i], :, :] = proportions[:, i, :, :]

        # with ground truth or learned distribution
        softmax = nn.Softmax(dim=1)(features)
        softmax_distribution = nn.AdaptiveAvgPool2d((1, 1))(softmax)
        positive_logits = nn.ReLU()(features)
        positive_logits = nn.AdaptiveAvgPool2d((1, 1))(positive_logits)
        positive_logits_distribution = positive_logits / positive_logits.sum(axis=1, keepdims=True)
        target_distribution = softmax_distribution if self.correction_mode == "softmax" else positive_logits_distribution
        dist_residual = distribution - target_distribution
        dist_residual = torch.where(distribution < 1e-6, torch.zeros_like(dist_residual), dist_residual) # don't apply updates when distribution label dne
        dist_residual_upsample = F.interpolate(dist_residual, size=softmax.size()[2:], mode="nearest")
        corrected_distribution = softmax + dist_residual_upsample if self.correction_mode == "softmax" else features + dist_residual_upsample

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

class PSPNetAggregation(nn.Module):
    """
    transfer learn pspnet with trained classification heads to benchmark classification head + presoftmax methods
    """
    def __init__(self, layers=50, classes=150, zoom_factor=8, pspnet_weights=None):
        super(PSPNetAggregation, self).__init__()
        self.pspnet = PSPNetContext(layers=layers, classes=classes, zoom_factor=zoom_factor, pspnet_weights=None)
        if pspnet_weights is not None:
            checkpoint = torch.load(pspnet_weights)['state_dict']
            fix_state_dict = OrderedDict()
            for k, v in checkpoint.items():
                name = k[7:] # remove 'module.' of dataparallel
                fix_state_dict[name] = v
            self.pspnet.load_state_dict(checkpoint, strict=False)
        for param in self.pspnet.parameters():
            param.requires_grad = False
        self.combo = DistributionMatch() # ContextLogitCombination()
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

        x = self.pspnet.pspnet.layer0(x)
        x = self.pspnet.pspnet.layer1(x)
        x = self.pspnet.pspnet.layer2(x)
        x = self.pspnet.pspnet.layer3(x)
        x = self.pspnet.pspnet.layer4(x)

        x = self.pspnet.pspnet.ppm(x)

        x = self.pspnet.pspnet.cls(x)
        
        x_alt, dist_pred = self.combo(x, distributions)

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