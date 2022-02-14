from distutils.command.config import config
from mmseg.apis import inference_segmentor, init_segmentor
import mmcv
import torch
from torch import nn
import torch.nn.functional as F
from mmseg.ops import resize

resnet_config = "exp/ade20k/upernet50/model/config.py"
resnet_checkpoint = "exp/ade20k/upernet50/model/upernet_r50.pth"
swin_config = "exp/ade20k/upernet_swin/model/config.py"
swin_checkpoint = "exp/ade20k/upernet_swin/model/upernet_swin_t.pth"

class UperNet(nn.Module):
    def __init__(self, backbone="resnet"):
        super(UperNet, self).__init__()
        assert backbone in ["resnet", "swin"]
        config = resnet_config if backbone == "resnet" else swin_config
        checkpoint = resnet_checkpoint if backbone == "resnet" else swin_checkpoint
        self.model = init_segmentor(config, checkpoint, device='cuda:0')
        self.backbone = self.model.backbone
        self.decode_head = self.model.decode_head

    def upernet_forward(self, inputs):
        inputs = self.decode_head._transform_inputs(inputs)

        # build laterals
        laterals = [
            lateral_conv(inputs[i])
            for i, lateral_conv in enumerate(self.decode_head.lateral_convs)
        ]

        laterals.append(self.decode_head.psp_forward(inputs))

        # build top-down path
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            prev_shape = laterals[i - 1].shape[2:]
            laterals[i - 1] = laterals[i - 1] + resize(
                laterals[i],
                size=prev_shape,
                mode='bilinear',
                align_corners=self.decode_head.align_corners)

        # build outputs
        fpn_outs = [
            self.decode_head.fpn_convs[i](laterals[i])
            for i in range(used_backbone_levels - 1)
        ]
        # append psp feature
        fpn_outs.append(laterals[-1])

        for i in range(used_backbone_levels - 1, 0, -1):
            fpn_outs[i] = resize(
                fpn_outs[i],
                size=fpn_outs[0].shape[2:],
                mode='bilinear',
                align_corners=self.decode_head.align_corners)
        fpn_outs = torch.cat(fpn_outs, dim=1)
        pre_cls = self.decode_head.fpn_bottleneck(fpn_outs)
        logits = self.decode_head.cls_seg(pre_cls)
        return pre_cls, logits

    def forward(self, x, y=None):
        h, w = x.size()[2:]
        x = self.backbone(x)  # list of tensors from stage1..4
        pre_cls, logits = self.upernet_forward(x)
        logits = F.interpolate(logits, size=(h,w), mode="bilinear", align_corners=self.decode_head.align_corners)
        return logits        


if __name__ == "__main__":
    x = torch.rand(size=(8, 3, 512, 512)).cuda()
    model = UperNet(backbone="swin")
    pred = model.forward(x)
    print("hello")