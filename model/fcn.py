import encoding
from torch import nn
import torch

class FCN(nn.Module):
    """
    wrapper around encnet module, putting it into terms similar to psp/psanet for convenience
    """
    def __init__(self, pretrained=True):
        super(FCN, self).__init__()
        self.model = encoding.models.get_model('FCN_ResNet50s_ADE', pretrained=pretrained)

    def forward(self, x, y=None):
        outs = self.model.forward(x)
        seg, aux = outs
        return seg, aux, None


if __name__ == "__main__":
    model = FCN(pretrained=True).to("cuda")
    x = torch.rand(size=(1, 3, 473, 473)).to("cuda")
    model.forward(x)
    print("Done!")