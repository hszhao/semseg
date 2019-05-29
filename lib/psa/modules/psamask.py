from torch import nn
from .. import functional as F


class PSAMask(nn.Module):
    def __init__(self, psa_type=0, mask_H_=None, mask_W_=None):
        super(PSAMask, self).__init__()
        assert psa_type in [0, 1]  # 0-col, 1-dis
        assert (mask_H_ in None and mask_W_ is None) or (mask_H_ is not None and mask_W_ is not None)
        self.psa_type = psa_type
        self.mask_H_ = mask_H_
        self.mask_W_ = mask_W_

    def forward(self, input):
        return F.psa_mask(input, self.psa_type, self.mask_H_, self.mask_W_)
