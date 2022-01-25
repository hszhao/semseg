import torch
from torch import nn
import torch.nn.functional as F

import model.resnet as models

def categorical_cross_entropy(y_pred, y_true, weights=None, smooth=0):
    """
    Weighted/Smoothed CCE
    """
    y_pred = torch.clamp(y_pred, 1e-10, 1 - 1e-10)
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

class ResNetDist(nn.Module):
    def __init__(self, layers=50, size=1, classes=150):
        super(ResNetDist, self).__init__()
        assert layers in [50, 101, 152]
        resnet = None
        if layers == 50:
            resnet = models.resnet50(pretrained=True)
        elif layers == 101:
            resnet = models.resnet101(pretrained=True)
        elif layers == 152:
            resnet = models.resnet152(pretrained=True)
        self.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.conv2, resnet.bn2, resnet.relu, resnet.conv3, resnet.bn3, resnet.relu, resnet.maxpool)
        self.layer1, self.layer2, self.layer3, self.layer4 = resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4

        for n, m in self.layer3.named_modules():
            if 'conv2' in n:
                m.dilation, m.padding, m.stride = (2, 2), (2, 2), (1, 1)
            elif 'downsample.0' in n:
                m.stride = (1, 1)
        for n, m in self.layer4.named_modules():
            if 'conv2' in n:
                m.dilation, m.padding, m.stride = (4, 4), (4, 4), (1, 1)
            elif 'downsample.0' in n:
                m.stride = (1, 1)

        fea_dim = 2048
        self.cls = nn.Sequential(
            nn.Conv2d(fea_dim, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((size, size)),
            nn.Dropout(p=0.2),
            nn.Conv2d(512, classes, kernel_size=1)
        )

        self.criterion = categorical_cross_entropy

    def forward(self, x, y=None):
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.cls(x)

        # x = nn.Softmax(dim=1)(x)
        x = nn.Sigmoid()(x)
        x = x / x.sum(dim=1, keepdims=True)

        if self.training:
            loss = self.criterion(x, y[1][0])
            return [x], loss
        else:
            return [x]