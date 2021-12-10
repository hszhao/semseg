import torch
from torch import nn
import torch.nn.functional as F

from collections import OrderedDict

# import sys, os
# sys.path.insert(0, os.path.abspath('.'))
from model.pspnet import PSPNet

class PPMClassification(nn.Module):
    def __init__(self, ppm):
        super(PPMClassification, self).__init__()
        self.ppm = ppm
        self.classification_head1 = nn.Conv2d(in_channels=512, out_channels=150, kernel_size=1)
        self.classification_head2 = nn.Conv2d(in_channels=512, out_channels=150, kernel_size=1)
        self.classification_head3 = nn.Conv2d(in_channels=512, out_channels=150, kernel_size=1)
        self.classification_head6 = nn.Conv2d(in_channels=512, out_channels=150, kernel_size=1)
        self.classification_heads = [self.classification_head1, self.classification_head2, self.classification_head3, self.classification_head6]

    def forward(self, x):
        x_size = x.size()
        out = [x]
        logits = []
        for f in range(len(self.ppm.features)):
            fn = self.ppm.features[f]
            # standard ppm
            out.append(F.interpolate(fn(x), x_size[2:], mode='bilinear', align_corners=True))
            # classification heads
            logits.append(self.classification_heads[f](fn(x)))
        # sigmoid activations
        classifications = [l.sigmoid() for l in logits]
        return torch.cat(out, 1), classifications, logits

class MultiPrediction(nn.Module):
    def __init__(self, cls):
        super(MultiPrediction, self).__init__()
        self.cls = cls
        self.classification_head = nn.Conv2d(in_channels=512, out_channels=150, kernel_size=(1,1), stride=(1,1))

    def forward(self, x):
        seg = self.cls(x) 
        final_feat = x
        for i in range(len(self.cls)):
            if i < len(self.cls) - 1:
                final_feat = self.cls[i](final_feat)
        
        # classification_feats = [
        #     nn.AdaptiveMaxPool2d((i, i))(final_feat) for i in [1, 2, 3, 6]
        # ]

        classification_feats = [
            nn.AdaptiveAvgPool2d((1, 1))(final_feat), final_feat
        ]

        logits = [
            self.classification_head(feat) for feat in classification_feats
        ]
        
        classifications = [nn.Sigmoid()(logit) for logit in logits]

        return seg, classifications, logits

class MultiDistribution(nn.Module):
    def __init__(self, cls):
        super(MultiDistribution, self).__init__()
        self.cls = cls
        # self.proj = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3,3), padding=1, bias=False)
        self.distribution_head = nn.Conv2d(in_channels=512, out_channels=150, kernel_size=(1,1), stride=(1,1))
        # self.distribution_head.weight.data = cls[-1].weight.data
        # self.distribution_head.bias.data = cls[-1].bias.data
        # self.act = nn.ReLU()
        # self.bn = nn.BatchNorm2d(512)

    def forward(self, x, scales):
        seg = self.cls(x) 
        final_feat = x
        for i in range(len(self.cls)):
            if i < len(self.cls) - 1:
                final_feat = self.cls[i](final_feat)

        softmax = nn.Softmax(dim=1)(seg)
        prediction_area = (softmax.size()[2]) * (softmax.size()[3])
        prediction_pool = nn.AvgPool2d(kernel_size=softmax.size()[3], divisor_override=1)(softmax)
        predicted_distribution = prediction_pool / prediction_area

        # pool_seg = nn.AdaptiveAvgPool2d((1,1))(final_feat)
        new_dist = self.distribution_head(final_feat)
        new_dist = nn.ReLU()(new_dist)
        # new_dist = nn.Softmax(dim=1)(new_dist)
        # new_dist = nn.ReLU()(new_dist)
        # new_dist = new_dist / new_dist.sum(axis=1, keepdims=True)
        new_dist = nn.AdaptiveAvgPool2d((1,1))(new_dist)
        new_dist = new_dist / (new_dist.sum(axis=1, keepdims=True))
        # new_dist = nn.ReLU()(new_dist) + 1e-9
        
        # weights = nn.Sigmoid()(weights)

        # softmax distribution baseline distance: 0.5166
        # predicted_distribution = predicted_distribution

        # + additional weights based on global average pool: 
        # predicted_distribution = predicted_distribution * weights
        # predicted_distribution = predicted_distribution / predicted_distribution.sum(axis=1, keepdims=True)
        return seg, [new_dist], 0

class MultiDistributionAE(nn.Module):
    def __init__(self, cls):
        super(MultiDistributionAE, self).__init__()
        self.cls = cls
        # self.proj = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3,3), padding=1, bias=False)
        self.enc1 = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=(1,1), stride=(1,1))
        self.enc2 = nn.Conv2d(in_channels=256, out_channels=150, kernel_size=(1,1), stride=(1,1))
        self.dec1 = nn.Conv2d(in_channels=150, out_channels=256, kernel_size=(1,1), stride=(1,1))
        self.dec2 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(1,1), stride=(1,1))
        self.distribution_head = nn.Conv2d(in_channels=150, out_channels=150, kernel_size=(1,1), stride=(1,1))
        self.act = nn.ReLU()
        self.bn1 = nn.BatchNorm2d(256)
        self.bn2 = nn.BatchNorm2d(150)
        self.bn3 = nn.BatchNorm2d(256)
        self.bn4 = nn.BatchNorm2d(512)
        self.sparsity = 1e-5

    def forward(self, x, scales):
        seg = self.cls(x) 
        final_feat = x
        for i in range(len(self.cls)):
            if i < len(self.cls) - 1:
                final_feat = self.cls[i](final_feat)

        # compress final feat
        enc = nn.ReLU()(self.bn1(self.enc1(final_feat)))
        latent = nn.ReLU()(self.bn2(self.enc2(enc)))
        dec = nn.ReLU()(self.bn3(self.dec1(latent)))
        dec = nn.ReLU()(self.bn4(self.dec2(dec)))

        recons_loss = F.mse_loss(final_feat, dec)
        # sparse_loss = self.sparsity * torch.mean(torch.abs(enc))
        ae_loss = recons_loss # + sparse_loss

        # use latent to build distribution
        new_dist = self.distribution_head(latent)
        new_dist = nn.ReLU()(new_dist)
        new_dist = nn.AdaptiveAvgPool2d((1,1))(new_dist)
        new_dist = new_dist / (new_dist.sum(axis=1, keepdims=True))
        # new_dist = nn.ReLU()(new_dist) + 1e-9
        
        # weights = nn.Sigmoid()(weights)

        # softmax distribution baseline distance: 0.5166
        # predicted_distribution = predicted_distribution

        # + additional weights based on global average pool: 
        # predicted_distribution = predicted_distribution * weights
        # predicted_distribution = predicted_distribution / predicted_distribution.sum(axis=1, keepdims=True)
        return seg, [new_dist], ae_loss

def categorical_cross_entropy(y_pred, y_true):
    y_pred = torch.clamp(y_pred, 1e-9, 1 - 1e-9)
    return -(y_true * torch.log(y_pred)).sum(dim=1).mean()

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
        self.aux_criterion = nn.KLDivLoss(log_target=False)
        self.pred = MultiDistribution(self.pspnet.cls)
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

        x = self.pspnet.ppm(x)

        x, distributions, ae_loss = self.pred(x, scales=[1])

        if self.pspnet.zoom_factor != 1:
            x = F.interpolate(x, size=(h, w), mode='bilinear', align_corners=True)

        if self.training:
            segmentation_label = y[0]
            distribution_labels = y[1]
            aux = self.pspnet.aux(x_tmp)
            if self.zoom_factor != 1:
                aux = F.interpolate(aux, size=(h, w), mode='bilinear', align_corners=True)
            main_loss = self.pspnet.criterion(x, segmentation_label)
            aux_loss = self.pspnet.criterion(aux, segmentation_label)
            distribution_loss = None
            for i in range(len(distributions)):
                if distribution_loss is None:
                    distribution_loss = categorical_cross_entropy(distributions[i][:,:,:,:], distribution_labels[i][:,:,:,:])
                else:
                    distribution_loss = distribution_loss + categorical_cross_entropy(distributions[i][:,:,:,:], distribution_labels[i][:,:,:,:])
            distribution_loss = distribution_loss / len(distributions)
            distribution_loss = distribution_loss # + ae_loss
            return x.max(1)[1], distributions, main_loss, aux_loss, distribution_loss
        else:
            return x, distributions
