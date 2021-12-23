from numpy import logical_and
import torch
from torch import nn
import torch.nn.functional as F

from collections import OrderedDict

import sys, os
sys.path.insert(0, os.path.abspath('.'))
from model.pspnet_c import PSPNetClassification
from util.classification_utils import extract_mask_distributions

class ClassificationLogitCombination(nn.Module):
    def __init__(self, classes=150, agg_dim=(6,6)):
        super(ClassificationLogitCombination, self).__init__()
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
    def __init__(self):
        super(DistributionMatch, self).__init__()

    def forward(self, features, predicted_distribution, gt_distribution=None):
        scale = predicted_distribution[0].shape[3]  # e.g. 1x1 .. 6x6
        softmax = nn.Softmax(dim=1)(features)
        softmax_distribution = nn.AdaptiveAvgPool2d((scale, scale))(softmax)
        correction = predicted_distribution[0] - softmax_distribution
        correction = F.interpolate(correction, size=softmax.size()[2:], mode="nearest")
        corrected_softmax = softmax + correction
        corrected_softmax = corrected_softmax / corrected_softmax.sum(axis=1, keepdims=True)

        corrected_softmax_gt = None
        if gt_distribution is not None:
            # METHOD 1 - APPLY UNIFORM CORRECTION ACROSS SOFTMAX MAP
            gt_correction = gt_distribution[0] - softmax_distribution
            gt_correction = F.interpolate(gt_correction, size=softmax.size()[2:], mode="nearest")
            corrected_softmax_gt = softmax + gt_correction
            corrected_softmax_gt = corrected_softmax_gt / corrected_softmax_gt.sum(axis=1, keepdims=True)

        return corrected_softmax, corrected_softmax_gt

class PSPNetAggregation(nn.Module):
    """
    transfer learn pspnet with trained classification heads to benchmark classification head + presoftmax methods
    """
    def __init__(self, layers=50, classes=150, zoom_factor=8, pspnet_weights=None):
        super(PSPNetAggregation, self).__init__()
        self.pspnet = PSPNetClassification(layers=layers, classes=classes, zoom_factor=zoom_factor, pspnet_weights=None)
        if pspnet_weights is not None:
            checkpoint = torch.load(pspnet_weights)['state_dict']
            # fix_state_dict = OrderedDict()
            # for k, v in checkpoint.items():
            #     name = k[7:] # remove 'module.' of dataparallel
            #     fix_state_dict[name] = v
            self.pspnet.load_state_dict(checkpoint)
        for param in self.pspnet.parameters():
            param.requires_grad = False
        self.combo = DistributionMatch() # ClassificationLogitCombination()
        self.zoom_factor = zoom_factor
        self.classes = classes

    def forward(self, x, y=None, distributions=None):
        x_size = x.size()
        assert (x_size[2]-1) % 8 == 0 and (x_size[3]-1) % 8 == 0
        h = int((x_size[2] - 1) / 8 * self.zoom_factor + 1)
        w = int((x_size[3] - 1) / 8 * self.zoom_factor + 1)

        x = self.pspnet.pspnet.layer0(x)
        x = self.pspnet.pspnet.layer1(x)
        x = self.pspnet.pspnet.layer2(x)
        x = self.pspnet.pspnet.layer3(x)
        x = self.pspnet.pspnet.layer4(x)

        x_tmp = self.pspnet.pspnet.ppm(x)

        x, distributions_pred, _ = self.pspnet.pred(x_tmp, scale=1)
        
        x_alt, x_alt_gt = self.combo(x, distributions_pred, [distributions])

        if self.zoom_factor != 1:
            x = F.interpolate(x, size=(h, w), mode='bilinear', align_corners=True)
            x_alt = F.interpolate(x_alt, size=(h, w), mode='bilinear', align_corners=True)
            x_alt_gt = F.interpolate(x_alt_gt, size=(h, w), mode='bilinear', align_corners=True)
        
        # softmax = nn.Softmax(dim=1)(x)
        # prediction_area = (softmax.size()[2]) * (softmax.size()[3])
        # prediction_pool = nn.AvgPool2d(kernel_size=softmax.size()[3], divisor_override=1)(softmax)
        # predicted_distribution = prediction_pool / prediction_area

        if self.training:
            segmentation_label = y
            main_loss = self.pspnet.pspnet.criterion(x_alt, segmentation_label)
            return x_alt.max(1)[1], main_loss
        else:
            return x, x_alt, x_alt_gt, distributions_pred, distributions