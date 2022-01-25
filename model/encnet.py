import encoding
from torch import nn
import torch
import gc

class EncNet(nn.Module):
    """
    wrapper around encnet module, putting it into terms similar to psp/psanet for convenience
    """
    def __init__(self, pretrained=True, correction=True, num_samples=5, dropout=0.3, drop_type="class_residuals", classes=150):
        super(EncNet, self).__init__()
        self.model = encoding.models.get_model('EncNet_ResNet50s_ADE', pretrained=pretrained)
        self.correction = correction
        self.num_samples = num_samples
        self.dropout = dropout
        # add one or two layers
        self.correction_layer_extra = nn.Linear(in_features=classes*4, out_features=classes*4, bias=True)
        self.correction_layer = nn.Linear(in_features=classes*4, out_features=classes, bias=True)
        self.criterion = nn.CrossEntropyLoss(ignore_index=255)
        self.model.pretrained.eval()
        self.drop_type = drop_type

    def forward(self, x, y=None):
        # standard forward pass
        seg, classification, aux = self.model.forward(x)
        seg = seg.detach()
        classification = classification.detach()
        aux = aux.detach()
        if not self.correction:
            return seg, aux, classification
        else:
            # dropout ~25% of inputs or resnet feature map
            # future work - this dropout could also target semantic clusters from initial prediction
            classfication_dist = torch.empty_like(classification).detach()
            segmentation_dist = torch.empty_like(classification).detach()
            cutout = torch.zeros_like(x).detach()
            if self.drop_type == "rand":
                for _ in range(self.num_samples):
                    rand = torch.rand_like(cutout).detach()
                    x_drop = torch.where(rand < self.dropout, cutout, x).detach()
                    seg_drop, classfication_drop, aux_drop = self.model.forward(x_drop)
                    seg_dist = nn.AdaptiveMaxPool2d(output_size=(1,1))(seg_drop) # 1 x C x 1 x 1 tensor
                    seg_dist = seg_dist.squeeze(2).squeeze(2) # 1 x C tensor
                    classfication_dist = torch.cat([classfication_drop, classfication_dist], dim=0).detach()
                    segmentation_dist = torch.cat([seg_dist, segmentation_dist], dim=0).detach()
            elif self.drop_type == "classes":
                # drop each class, collect statistics over entire set of images
                predicted_mask = seg.max(1)[1].float().unsqueeze(1).detach()
                class_predictions = torch.unique(predicted_mask)
                for class_ind in range(class_predictions.size()[0]):
                    drop_class = (class_predictions[class_ind].float() * torch.ones_like(x)).detach()
                    x_drop = torch.where(predicted_mask == drop_class, cutout, x).detach()
                    seg_drop, classfication_drop, aux_drop = self.model.forward(x_drop)
                    aux_drop = aux_drop.detach()
                    seg_drop = seg_drop.detach()
                    seg_dist = nn.AdaptiveMaxPool2d(output_size=(1,1))(seg_drop) # 1 x C x 1 x 1 tensor
                    seg_dist = seg_dist.squeeze(2).squeeze(2) # 1 x C tensor
                    classfication_dist = torch.cat([classfication_drop, classfication_dist], dim=0).detach()
                    segmentation_dist = torch.cat([seg_dist, segmentation_dist], dim=0).detach()
            elif self.drop_type == "class_residuals":
                # drop each class, collect residual for each independent class w.r.t. original prediction
                predicted_mask = seg.max(1)[1].float().unsqueeze(1).detach()
                class_predictions = torch.unique(predicted_mask).detach()
                residual = torch.zeros_like(classification).detach()
                for class_ind in range(class_predictions.size()[0]):
                    drop_class = (class_predictions[class_ind].float() * torch.ones_like(x)).detach()
                    x_drop = torch.where(predicted_mask == drop_class, cutout, x).detach()
                    seg_drop, classification_drop, aux_drop = self.model.forward(x_drop)
                    aux_drop = aux_drop.detach()
                    seg_drop = seg_drop.detach()
                    classification_drop = classification_drop.detach()
                    class_residual = classification - classification_drop
                    residual[:, class_ind] = class_residual[:, class_ind]
                return seg, aux, classification, residual
            classification_mu = torch.mean(classfication_dist, dim=0, keepdim=True)
            segmentation_mu = torch.mean(segmentation_dist, dim=0, keepdim=True)
            classification_std = torch.std(classfication_dist, dim=0, keepdim=True)
            segmentation_std = torch.std(segmentation_dist, dim=0, keepdim=True)
            context = torch.cat([classification_mu, segmentation_mu, classification_std, segmentation_std], dim=-1)
            context = self.correction_layer_extra(context)
            context_reweighting = nn.Sigmoid()(self.correction_layer(context)) # 1 x C weights
            context_reweighting = torch.unsqueeze(torch.unsqueeze(context_reweighting, dim=-1), dim=-1)  # 1 x C x 1 x 1 weights 
            seg_corrected = seg * context_reweighting
            if y is not None:
                loss = self.criterion(seg_corrected, y)
                return seg, seg_corrected, classification, loss
            else:
                return seg_corrected, aux, classification

    def pretrained_forward(self, x):
        """
        Pass forward ResNet, get final feature maps for each scale
        """
        x = self.model.pretrained.conv1(x)
        x = self.model.pretrained.bn1(x)
        x = self.model.pretrained.relu(x)
        x = self.model.pretrained.maxpool(x)
        layer1 = self.model.pretrained.layer1(x)
        layer2 = self.model.pretrained.layer2(x)
        layer3 = self.model.pretrained.layer3(x)
        layer4 = self.model.pretrained.layer4(x)
        return layer1, layer2, layer3, layer4


if __name__ == "__main__":
    model = EncNet(pretrained=True).to("cuda")
    x = torch.rand(size=(4, 3, 473, 473)).to("cuda")
    model.forward(x)
    print("Done!")