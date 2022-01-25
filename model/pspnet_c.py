from typing import final
import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F

from collections import OrderedDict

# import sys, os
# sys.path.insert(0, os.path.abspath('.'))
from model.pspnet import PSPNet

import pandas as pd
info = pd.read_csv("dataset/ade20k/objectInfo150.txt", sep="\t")
classes = info["Name"].values
classes_short = [c.split(",")[0] for c in classes]

dist = info["Ratio"].values

class PPMContext(nn.Module):
    def __init__(self, ppm):
        super(PPMContext, self).__init__()
        self.ppm = ppm
        self.classification_head = nn.Conv2d(in_channels=512, out_channels=150, kernel_size=1)
        self.distribution_head = nn.Conv2d(in_channels=512, out_channels=150, kernel_size=1)
        self.dropout = nn.Dropout2d(p=0.2)

    def forward(self, x):
        x_size = x.size()
        out = [x]
        classifications = []
        distributions = []
        for f in range(len(self.ppm.features)):
            fn = self.ppm.features[f]
            # standard ppm
            out.append(F.interpolate(fn(x), x_size[2:], mode='bilinear', align_corners=True))
            # context heads
            cls = nn.Sigmoid()(self.classification_head(self.dropout(fn(x))))
            dist = nn.Softmax(dim=1)(self.distribution_head(self.dropout(fn(x))) * cls)
            classifications.append(cls)
            distributions.append(dist)

        return torch.cat(out, 1), classifications, distributions

class ContextPrediction(nn.Module):
    def __init__(self, cls):
        super(ContextPrediction, self).__init__()
        self.cls = cls
        self.classification_head = nn.Conv2d(in_channels=512, out_channels=150, kernel_size=1)
        self.distribution_head = nn.Conv2d(in_channels=512, out_channels=150, kernel_size=1)
        self.dropout = nn.Dropout2d(p=0.2)

    def forward(self, x):
        # single scale only for now       
        for i in range(len(self.cls)-1):
            x = self.cls[i](x)
        
        segmentation = self.cls[-1](x)
        pool = nn.AdaptiveAvgPool2d(output_size=(1,1))(x)
        classification = nn.Sigmoid()(self.classification_head(pool))
        distribution = nn.Softmax(dim=1)(self.distribution_head(pool) * classification)

        return segmentation, [classification], [distribution]


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
    def __init__(self, layers=50, classes=150, zoom_factor=8, pspnet_weights=None, alt_pyramid=False, alt_cls=False):
        super(PSPNetContext, self).__init__()
        self.pspnet = PSPNet(layers=layers, classes=classes, zoom_factor=zoom_factor, pretrained=False)
        if pspnet_weights is not None:
            checkpoint = torch.load(pspnet_weights)['state_dict']
            fix_state_dict = OrderedDict()
            for k, v in checkpoint.items():
                name = k[7:] # remove 'module.' of dataparallel
                fix_state_dict[name] = v
            self.pspnet.load_state_dict(fix_state_dict)
        for param in self.pspnet.parameters():
            param.requires_grad = False
        self.class_criterion = nn.BCELoss()
        self.alt_pyramid = alt_pyramid
        self.alt_cls = alt_cls
        self.pyramid = None
        self.prediction = None
        if alt_pyramid:
            self.pyramid = PPMContext(self.pspnet.ppm)
        if alt_cls:
            self.prediction = ContextPrediction(self.pspnet.cls)
        self.zoom_factor = zoom_factor
        self.classes = classes

    def forward(self, x, y=None):
        x_size = x.size()
        assert (x_size[2]-1) % 8 == 0 and (x_size[3]-1) % 8 == 0
        h = int((x_size[2] - 1) / 8 * self.zoom_factor + 1)
        w = int((x_size[3] - 1) / 8 * self.zoom_factor + 1)

        classifications, distributions = None, None

        x = self.pspnet.layer0(x)
        x = self.pspnet.layer1(x)
        x = self.pspnet.layer2(x)
        x_tmp = self.pspnet.layer3(x)
        x = self.pspnet.layer4(x_tmp)

        if not self.alt_pyramid:
            x = self.pspnet.ppm(x)
        else:
            x, classifications, distributions = self.pyramid(x)
        
        if not self.alt_cls:
            x = self.pspnet.cls(x)
        else:
            x, classifications, distributions = self.prediction(x)

        if self.pspnet.zoom_factor != 1:
            x = F.interpolate(x, size=(h, w), mode='bilinear', align_corners=True)

        if self.training:
            segmentation_label = y[0]
            distribution_labels = y[1]
            class_labels = y[2]
            # aux = self.pspnet.aux(x_tmp)
            # if self.zoom_factor != 1:
            #     aux = F.interpolate(aux, size=(h, w), mode='bilinear', align_corners=True)
            # aux_loss = self.pspnet.criterion(aux, segmentation_label)
            main_loss = self.pspnet.criterion(x, segmentation_label)
            distribution_loss = None
            classification_loss = None
            for i in range(len(distribution_labels)):
                if distribution_loss is None:
                    distribution_loss = categorical_cross_entropy(distributions[i], distribution_labels[i])
                else:
                    distribution_loss = distribution_loss + categorical_cross_entropy(distributions[i], distribution_labels[i])
                if classification_loss is None:
                    classification_loss = nn.BCELoss()(classifications[i], class_labels[i])
                else:
                    classification_loss = classification_loss + nn.BCELoss()(classifications[i], class_labels[i])
            distribution_loss = distribution_loss / len(distribution_labels)
            classification_loss  = classification_loss / len(distribution_labels)
            return x.max(1)[1], distributions, classifications, main_loss, distribution_loss, classification_loss
        else:
            return x, distributions, classifications
