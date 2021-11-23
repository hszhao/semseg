import torch
from torch import nn
import torch.nn.functional as F

from collections import OrderedDict

# import sys, os
# sys.path.insert(0, os.path.abspath('.'))
from model.pspnet_c import PSPNetClassification

class ClassificationAttentionConv(nn.Module):
    def __init__(self, classes=150, agg_dim=(6,6)):
        super(ClassificationAttentionConv, self).__init__()
        self.classes=150
        self.agg_dim = agg_dim
        self.combination1 = nn.Conv2d(in_channels=classes*4, out_channels=classes, kernel_size=1, bias=False)
        self.combination2 = nn.Conv2d(in_channels=classes, out_channels=classes, padding=1, kernel_size=3, bias=False)

    def forward(self, x, out_size):
        # upsample all scales to agg dim
        upsampled = []
        for scale in x:
            upsampled.append(F.interpolate(scale, size=self.agg_dim, mode='nearest'))
        concat = torch.cat(upsampled, dim=1)
        channel_weights = self.combination1(concat)
        channel_weights = self.combination2(channel_weights)
        channel_weights = F.interpolate(channel_weights, size=out_size, mode='bilinear', align_corners=True)
        channel_weights = torch.sigmoid(channel_weights)
        return channel_weights

class ClassificationBiasConv(nn.Module):
    def __init__(self, classes=150, agg_dim=(6,6)):
        super(ClassificationBiasConv, self).__init__()
        self.classes=150
        self.agg_dim = agg_dim
        self.combination1 = nn.Conv2d(in_channels=classes*4, out_channels=classes, kernel_size=1, bias=False)
        self.combination2 = nn.Conv2d(in_channels=classes, out_channels=classes, padding=1, kernel_size=3, bias=False)
        self.combination3 = nn.Conv2d(in_channels=classes*2, out_channels=classes, kernel_size=1, bias=False)
        orig_weights = torch.eye(classes)
        orig_weights = torch.unsqueeze(orig_weights, -1)
        orig_weights = torch.unsqueeze(orig_weights, -1)
        classif_weights = torch.zeros((150, 150, 1, 1))
        self.combination3.weight.data = torch.cat([orig_weights, classif_weights], dim=1)

    def forward(self, features, classifications, out_size):
        # upsample all scales to agg dim
        upsampled = []
        for scale in classifications:
            upsampled.append(F.interpolate(scale, size=self.agg_dim, mode='nearest'))
        concat = torch.cat(upsampled, dim=1)
        channel_weights = self.combination1(concat)
        channel_weights = self.combination2(channel_weights)
        channel_weights = F.interpolate(channel_weights, size=out_size, mode='bilinear', align_corners=True)
        channel_weights = torch.sigmoid(channel_weights)
        concat2 = torch.cat([features, channel_weights], dim=1)
        reweighted_features = self.combination3(concat2)
        return reweighted_features

class ClassificationAttentionMlp(nn.Module):
    def __init__(self, classes=150, agg_dim=(6,6)):
        super(ClassificationAttentionMlp, self).__init__()
        self.classes=150
        self.agg_dim = agg_dim
        in_dim = (1 + 2**2 + 3**2 + 6**2)*classes
        self.combination1 = nn.Linear(in_features=in_dim, out_features=classes*6*6, bias=False)

    def forward(self, x, out_size):
        # upsample all scales to agg dim
        flattened = []
        for scale in x:
            flattened.append(torch.flatten(scale, start_dim=1))
        concat = torch.cat(flattened, dim=1)
        channel_weights = self.combination1(concat)
        channel_weights = torch.reshape(channel_weights, shape=((-1, self.classes, self.agg_dim[0], self.agg_dim[1])))
        channel_weights = F.interpolate(channel_weights, size=out_size, mode='bilinear', align_corners=True)
        channel_weights = torch.sigmoid(channel_weights)
        return channel_weights

class PSPNetAggregation(nn.Module):
    """
    transfer learn pspnet with trained classification heads to benchmark classification head + presoftmax methods
    """
    def __init__(self, layers=50, classes=150, zoom_factor=8, agg="conv_conv", pspnet_weights=None):
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
        # self.channel_weights = ClassificationAttentionConv() if agg == "conv" else ClassificationAttentionMlp()
        self.channel_weights = ClassificationBiasConv()
        self.zoom_factor = zoom_factor
        self.classes = classes

    def forward(self, x, y=None, classifications=None):
        x_size = x.size()
        assert (x_size[2]-1) % 8 == 0 and (x_size[3]-1) % 8 == 0
        h = int((x_size[2] - 1) / 8 * self.zoom_factor + 1)
        w = int((x_size[3] - 1) / 8 * self.zoom_factor + 1)

        x = self.pspnet.pspnet.layer0(x)
        x = self.pspnet.pspnet.layer1(x)
        x = self.pspnet.pspnet.layer2(x)
        x_tmp = self.pspnet.pspnet.layer3(x)
        x = self.pspnet.pspnet.layer4(x_tmp)

        x, class_pred = self.pspnet.ppm_classif(x)

        if classifications is None:
            classifications = class_pred

        # weights = self.channel_weights(classifications, out_size=x.size()[2:])

        x = self.pspnet.pspnet.cls(x)

        x = self.channel_weights(features=x, classifications=classifications, out_size=x.size()[2:])

        # x = x * weights

        if self.zoom_factor != 1:
            x = F.interpolate(x, size=(h, w), mode='bilinear', align_corners=True)

        if self.training:
            segmentation_label = y
            main_loss = self.pspnet.pspnet.criterion(x, segmentation_label)
            return x.max(1)[1], main_loss
        else:
            return x, classifications

# if __name__ == '__main__':
#     import os
#     os.environ["CUDA_VISIBLE_DEVICES"] = '0, 1'
#     input = torch.rand(4, 3, 473, 473).cuda()
#     model = PSPNet(layers=50, bins=(1, 2, 3, 6), dropout=0.1, classes=21, zoom_factor=1, use_ppm=True, pretrained=True).cuda()
#     model.eval()
#     print(model)
#     output = model(input)
#     print('PSPNet', output.size())