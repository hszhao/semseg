# "pyramid encoding network"
# simplifications: using GAP rather than encoding layer
# TODO revisit; not currently used in experiments

from typing import final
import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F

from collections import OrderedDict

import sys, os
sys.path.insert(0, os.path.abspath('.'))
from model.pspnet import PSPNet

class PPMContext(nn.Module):
    """
    shared classification heads across all scales of pyramid
    """
    def __init__(self, ppm, cls):
        """
        constructor input: 
            - ppm module (4 scale pooling+conv)
            - cls module (2048->512, 512->C)
        """
        super(PPMContext, self).__init__()
        self.ppm = ppm
        self.cls = cls
        self.conv1 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(512)
        self.conv2 = nn.Conv2d(in_channels=2048, out_channels=512, kernel_size=1)
        self.class_head = nn.Conv2d(in_channels=512, out_channels=150, kernel_size=1)
        self.max_pool_size = 6

    def forward(self, x):
        x_size = x.size()
        out = [x]
        features = []
        classifications = []
        for f in self.ppm.features:
            # feature --> classification
            feature = nn.ReLU()(self.bn(self.conv1(f(x))))
            classification = nn.Sigmoid()(self.class_head(feature))
            classifications.append(classification)
            # upsample features
            features.append(F.interpolate(feature, (self.max_pool_size), mode='bilinear', align_corners=True))
            
        features = torch.cat(features, 1) # Nx6x6x2028
        features = self.conv2(features) # Nx6x6x512
        features = nn.AdaptiveAvgPool2d(output_size=(1,1))(features) # Nx1x1x512
        channel_weights = nn.Sigmoid()(features) # Nx1x1x512 channel weights

        # regular ppm forward pass
        for f in self.ppm.features:
            out.append(F.interpolate(f(x), x_size[2:], mode='bilinear', align_corners=True))
        
        combined_features = torch.cat(out, 1)  # NxWxHx2048

        # final prediction (original)
        # self.cls = nn.Sequential(
        #     nn.Conv2d(fea_dim, 512, kernel_size=3, padding=1, bias=False),
        #     nn.BatchNorm2d(512),
        #     nn.ReLU(inplace=True),
        #     nn.Dropout2d(p=dropout),
        #     nn.Conv2d(512, classes, kernel_size=1)
        # )
        reduced_features = self.cls[0](combined_features)
        reduced_features = self.cls[1](reduced_features)
        reduced_features = self.cls[2](reduced_features)
        reduced_features = reduced_features * channel_weights  # added channel weight mechanism
        reduced_features = self.cls[3](reduced_features)

        # SEGMENTATION + (DISTRIBUTION HEADS)
        segmentation = self.cls[4](reduced_features)

        return segmentation, classifications

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

class PyramidContextNetwork(nn.Module):
    def __init__(self, layers=50, classes=150, zoom_factor=8, pspnet_weights=None, refit_pspnet=False):
        super(PyramidContextNetwork, self).__init__()
        self.pspnet = PSPNet(layers=layers, classes=classes, zoom_factor=zoom_factor, pretrained=True)
        if pspnet_weights is not None:
            checkpoint = torch.load(pspnet_weights)['state_dict']
            fix_state_dict = OrderedDict()
            for k, v in checkpoint.items():
                name = k[7:] # remove 'module.' of dataparallel
                fix_state_dict[name] = v
            self.pspnet.load_state_dict(fix_state_dict)
        if not refit_pspnet:
            for param in self.pspnet.parameters():
                param.requires_grad = False
        self.aux_criterion = nn.CrossEntropyLoss(ignore_index=255)
        self.pyramid = PPMContext(self.pspnet.ppm, self.pspnet.cls)
        self.zoom_factor = zoom_factor
        self.classes = classes

    def forward(self, x, y=None):
        """
        Note that our y will have several components:

        y[0] target segmentation mask
        y[1] list of classification targets at each scale in the pyramid,
            Nx1x1xC, Nx2x2xC, etc...
        """
        x_size = x.size()
        assert (x_size[2]-1) % 8 == 0 and (x_size[3]-1) % 8 == 0
        h = int((x_size[2] - 1) / 8 * self.zoom_factor + 1)
        w = int((x_size[3] - 1) / 8 * self.zoom_factor + 1)

        x = self.pspnet.layer0(x)
        x = self.pspnet.layer1(x)
        x = self.pspnet.layer2(x)
        x_tmp = self.pspnet.layer3(x)
        x = self.pspnet.layer4(x_tmp)

        x, classifications = self.pyramid(x)

        if self.pspnet.zoom_factor != 1:
            x = F.interpolate(x, size=(h, w), mode='bilinear', align_corners=True)

        # MANY LOSSES HERE
        # Auxiliary and Main Segmentation Loss
        # Multi-Scale Classification Loss
        # (Distribution Loss) - TODO
        if y is not None:
            segmentation_label = y[0]
            class_labels = y[1]
            aux = self.pspnet.aux(x_tmp)
            if self.zoom_factor != 1:
                aux = F.interpolate(aux, size=(h, w), mode='bilinear', align_corners=True)
            main_loss = self.pspnet.criterion(x, segmentation_label)
            aux_loss = self.pspnet.criterion(aux, segmentation_label)
            class_loss = None
            for i in range(1):
                if class_loss is None:
                    class_loss = nn.BCELoss()(classifications[i], class_labels[i])
                else:
                    class_loss += nn.BCELoss()(classifications[i], class_labels[i])
            class_loss = class_loss / len(classifications)
            return x.max(1)[1], classifications, main_loss, aux_loss, class_loss
        else:
            return x, classifications



if __name__ == "__main__":
    model = PyramidContextNetwork().to("cuda")
    x = torch.rand(size=(4, 3, 473, 473)).to("cuda")
    model.forward(x)
    print("Done!")