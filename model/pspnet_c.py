import torch
from torch import nn
import torch.nn.functional as F

from collections import OrderedDict

# import sys, os
# sys.path.insert(0, os.path.abspath('.'))
from model.pspnet import PSPNet

class PPMClassification(nn.Module):
    def __init__(self, ppm, classification_head, residual=True):
        super(PPMClassification, self).__init__()
        self.ppm = ppm
        self.classification_head = classification_head
        self.residual = residual

    def forward(self, x):
        x_size = x.size()
        out = [x]
        classifications = []
        prev_scale_class = None
        for f in self.ppm.features:
            # standard ppm
            out.append(F.interpolate(f(x), x_size[2:], mode='bilinear', align_corners=True))
            # classification heads
            if self.residual:
                if prev_scale_class is None:
                    # start with first scale alone
                    prev_scale_class = self.classification_head(f(x))
                    classifications.append(prev_scale_class)
                else:
                    # residual prediction w.r.t. previous scale
                    classification = F.interpolate(prev_scale_class, size=f(x).size()[2:], mode="bilinear", align_corners=False) + self.classification_head(f(x))
                    prev_scale_class = classification
                    classifications.append(self.classification_head(f(x)))
            else:
                classifications.append(prev_scale_class)
        # sigmoid activations
        classifications = [c.sigmoid() for c in classifications]
        return torch.cat(out, 1), classifications

class PSPNetClassification(nn.Module):
    def __init__(self, layers=50, classes=150, zoom_factor=8, pspnet_weights=None):
        super(PSPNetClassification, self).__init__()
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
        self.classification_head = nn.Conv2d(in_channels=512, out_channels=classes, kernel_size=1)
        self.classification_criterion = torch.nn.BCELoss()    
        self.ppm_classif = PPMClassification(self.pspnet.ppm, self.classification_head)
        self.zoom_factor = zoom_factor
        self.classes = classes

    def forward(self, x, y=None):
        x_size = x.size()
        assert (x_size[2]-1) % 8 == 0 and (x_size[3]-1) % 8 == 0
        h = int((x_size[2] - 1) / 8 * self.zoom_factor + 1)
        w = int((x_size[3] - 1) / 8 * self.zoom_factor + 1)

        x = self.pspnet.layer0(x)
        x = self.pspnet.layer1(x)
        x = self.pspnet.layer2(x)
        x_tmp = self.pspnet.layer3(x)
        x = self.pspnet.layer4(x_tmp)

        x, classifications = self.ppm_classif(x)

        x = self.pspnet.cls(x)
        if self.pspnet.zoom_factor != 1:
            x = F.interpolate(x, size=(h, w), mode='bilinear', align_corners=True)

        if self.training:
            segmentation_label = y[0]
            classification_labels = y[1]
            aux = self.pspnet.aux(x_tmp)
            if self.zoom_factor != 1:
                aux = F.interpolate(aux, size=(h, w), mode='bilinear', align_corners=True)
            main_loss = self.pspnet.criterion(x, segmentation_label)
            aux_loss = self.pspnet.criterion(aux, segmentation_label)
            classification_loss = None
            for i in range(4):
                for c in range(self.classes):
                    if classification_loss is None:
                        classification_loss = self.classification_criterion(classifications[i][:,c,:,:], classification_labels[i][:,c,:,:])
                    else:
                        classification_loss = classification_loss + self.classification_criterion(classifications[i][:,c,:,:], classification_labels[i][:,c,:,:])
            return x.max(1)[1], classifications, main_loss, aux_loss, classification_loss
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