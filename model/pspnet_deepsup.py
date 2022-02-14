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

class SupHead(nn.Module):
    """
    Residual Block --> (GAP) --> Segment/Cls Heads
    """
    def __init__(self, in_features, out_features=512, classes=150):
        super(SupHead, self).__init__()
        self.shortcut = nn.Conv2d(in_features, out_features, kernel_size=1, bias=False)
        self.bn_shortcut = nn.BatchNorm2d(out_features)
        self.conv1 = nn.Conv2d(out_features, out_features, kernel_size=3, padding=1, bias=False)
        self.conv2 = nn.Conv2d(out_features, out_features, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_features)
        self.bn2 = nn.BatchNorm2d(out_features)
        self.seg = nn.Conv2d(out_features, classes, kernel_size=1, bias=True)
        self.cls = nn.Conv2d(out_features, classes, kernel_size=1, bias=True)

    def forward(self, x):
        residual = self.shortcut(x)
        residual = self.bn_shortcut(residual)
        x = self.conv1(residual)
        x = self.bn1(x)
        x = nn.ReLU()(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = residual + x
        x = nn.ReLU()(x)
        seg_pred = self.seg(x)
        pool = nn.AdaptiveAvgPool2d((1, 1))(x)
        class_pred = nn.Softmax(dim=1)(self.cls(pool))
        return x, seg_pred, class_pred

class PSPNetDeepSup(nn.Module):
    """
    transfer learn pspnet with trained classification heads to benchmark classification head + presoftmax methods
    """
    def __init__(self, layers=50, classes=150, zoom_factor=8, pspnet_weights=None):
        super(PSPNetDeepSup, self).__init__()
        self.pspnet = PSPNet(layers=layers, classes=classes, zoom_factor=zoom_factor, pretrained=True)
        if pspnet_weights is not None:
            checkpoint = torch.load(pspnet_weights)['state_dict']
            fix_state_dict = OrderedDict()
            for k, v in checkpoint.items():
                name = k[7:] # remove 'module.' of dataparallel
                fix_state_dict[name] = v
            self.pspnet.load_state_dict(fix_state_dict, strict=True)
        for param in self.pspnet.parameters():
            param.requires_grad = False
        self.loss = categorical_cross_entropy
        self.zoom_factor = zoom_factor
        self.classes = classes
        self.l1_sup = SupHead(in_features=256)
        self.l2_sup = SupHead(in_features=512)
        self.l3_sup = SupHead(in_features=1024)
        self.l4_sup = SupHead(in_features=2048)

    def forward(self, x, y=None):
        """
        y[0] = segmentation label
        y[1] = class label
        distributions = GT distributions for inference
        """
        segmentation_label = y[0] if y is not None else y
        class_label = y[1] if y is not None else y

        x_size = x.size()
        assert (x_size[2]-1) % 8 == 0 and (x_size[3]-1) % 8 == 0
        h = int((x_size[2] - 1) / 8 * self.zoom_factor + 1)
        w = int((x_size[3] - 1) / 8 * self.zoom_factor + 1)

        x = self.pspnet.layer0(x)
        x1 = self.pspnet.layer1(x)
        x2 = self.pspnet.layer2(x1)
        x3 = self.pspnet.layer3(x2)
        x4 = self.pspnet.layer4(x3)

        final_feat = self.pspnet.ppm(x4)
        for i in range(len(self.pspnet.cls)-2):
            final_feat = self.pspnet.cls[i](final_feat)  # final pre-seg features
        
        x = self.pspnet.cls[-2](final_feat)  # dropout
        x = self.pspnet.cls[-1](x)  # segmentation logits     
        if self.zoom_factor != 1:
            x = F.interpolate(x, size=(h, w), mode='bilinear', align_corners=True)

        if self.training:
            l1, s1, c1 = self.l1_sup(x1)
            del x1
            l2, s2, c2 = self.l2_sup(x2)
            del x2
            l3, s3, c3 = self.l3_sup(x3)
            del x3
            l4, s4, c4 = self.l4_sup(x4)
            del x4
            torch.cuda.empty_cache()

            seg_losses = 0
            seg_dist = nn.AdaptiveAvgPool2d((1,1))(nn.Softmax(dim=1)(x)).detach()
            for seg in [s1, s2, s3, s4]:            
                seg = nn.AdaptiveAvgPool2d((1,1))(nn.Softmax(dim=1)(seg))
                seg_losses += categorical_cross_entropy(seg, seg_dist)
            seg_losses /= 4

            del s1
            del s2
            del s3
            del s4
            torch.cuda.empty_cache()

            class_losses = 0
            for cls in [c1, c2, c3, c4]:
                class_losses += categorical_cross_entropy(cls, class_label)
            class_losses /= 4
            
            hint_losses = 0
            # target_feat = final_feat.detach()
            # for feat in [l1, l2, l3, l4]:
            #     feat = F.interpolate(feat, size=(60, 60), mode='bilinear', align_corners=True)
            #     hint_losses += torch.mean(torch.square(feat - target_feat))
            # hint_losses /= 4

            del l1
            del l2
            del l3
            del l4
            torch.cuda.empty_cache()

            main_loss = self.pspnet.criterion(x, segmentation_label) 
            return x.max(1)[1], main_loss, seg_losses, class_losses# , hint_losses
        else:
            return x
