import torch
from torch.autograd import Function
from .. import src


class PSAMask(Function):
    def __init__(self, psa_type=0, mask_H_=None, mask_W_=None):
        super(PSAMask, self).__init__()
        assert psa_type in [0, 1]  # 0-col, 1-dis
        self.psa_type = psa_type
        assert (mask_H_ is None and mask_W_ is None) or (mask_H_ is not None and mask_W_ is not None)
        self.mask_H_ = mask_H_
        self.mask_W_ = mask_W_

    def forward(self, input):
        num_, channels_, feature_H_, feature_W_ = input.size()
        if self.mask_H_ is not None and self.mask_W_ is not None:
            mask_H_ = self.mask_H_
            mask_W_ = self.mask_W_
            assert mask_H_ % 2 == 1
            assert mask_W_ % 2 == 1
        else:
            mask_H_ = 2 * feature_H_ - 1
            mask_W_ = 2 * feature_W_ - 1
        assert channels_ == mask_H_ * mask_W_
        half_mask_H_ = (mask_H_ - 1) // 2
        half_mask_W_ = (mask_W_ - 1) // 2
        output = torch.zeros([num_, feature_H_ * feature_W_, feature_H_, feature_W_], dtype=input.dtype, device=input.device)
        ''' python implementation - slow
        for n in range(num_):
            for h in range(feature_H_):
                for w in range(feature_W_):
                    hstart = max(0, half_mask_H_ - h)
                    hend = min(mask_H_, feature_H_ + half_mask_H_ - h)
                    wstart = max(0, half_mask_W_ - w)
                    wend = min(mask_W_, feature_W_ + half_mask_W_ - w)
                    hstart_com = max(0, h - half_mask_H_)
                    hend_com = min(feature_H_, h + half_mask_H_ + 1)
                    wstart_com = max(0, w - half_mask_W_)
                    wend_com = min(feature_W_, w + half_mask_W_ + 1)
                    mask_ori = input[n, :, h, w].view(mask_H_, mask_W_)
                    mask_com = torch.zeros(feature_H_, feature_W_, dtype=input.dtype, device=input.device)
                    mask_com[hstart_com:hend_com, wstart_com:wend_com] = mask_ori[hstart:hend, wstart:wend]
                    if self.psa_type == 0:  # col
                        output[n, :, h, w] = mask_com.view(-1)
                    else:  # dis
                        c = h * feature_W_ + w
                        output[n, c, :, :] = mask_com.view(feature_H_, feature_W_)
        if self.psa_type == 1:  # dis
            output = output.view(num_, feature_H_ * feature_W_, feature_H_ * feature_W_).transpose(1, 2).view(num_, feature_H_ * feature_W_, feature_H_, feature_W_)
        '''
        # c/cuda implementation
        if not input.is_cuda:
            src.cpu.psamask_forward(self.psa_type, input, output, num_, feature_H_, feature_W_, mask_H_, mask_W_, half_mask_H_, half_mask_W_)
        else:
            output = output.cuda()
            src.gpu.psamask_forward(self.psa_type, input, output, num_, feature_H_, feature_W_, mask_H_, mask_W_, half_mask_H_, half_mask_W_)
        self.num_, self.channels_, self.feature_H_, self.feature_W_, self.mask_H_, self.mask_W_, self.half_mask_H_, self.half_mask_W_ = num_, channels_, feature_H_, feature_W_, mask_H_, mask_W_, half_mask_H_, half_mask_W_
        return output

    def backward(self, grad_output):
        num_, channels_, feature_H_, feature_W_, mask_H_, mask_W_, half_mask_H_, half_mask_W_ = self.num_, self.channels_, self.feature_H_, self.feature_W_, self.mask_H_, self.mask_W_, self.half_mask_H_, self.half_mask_W_
        grad_input = torch.zeros([num_, channels_, feature_H_, feature_W_], dtype=grad_output.dtype, device=grad_output.device)
        ''' python implementation - slow
        if self.psa_type == 1:  # dis
            grad_output = grad_output.view(num_, feature_H_ * feature_W_, feature_H_ * feature_W_).transpose(1, 2).view(num_, feature_H_ * feature_W_, feature_H_, feature_W_)
        for n in range(num_):
            for h in range(feature_H_):
                for w in range(feature_W_):
                    hstart = max(0, half_mask_H_ - h)
                    hend = min(mask_H_, feature_H_ + half_mask_H_ - h)
                    wstart = max(0, half_mask_W_ - w)
                    wend = min(mask_W_, feature_W_ + half_mask_W_ - w)
                    hstart_com = max(0, h - half_mask_H_)
                    hend_com = min(feature_H_, h + half_mask_H_ + 1)
                    wstart_com = max(0, w - half_mask_W_)
                    wend_com = min(feature_W_, w + half_mask_W_ + 1)
                    if self.psa_type == 0:  # col
                        grad_mask_com = grad_output[n, :, h, w].view(feature_H_, feature_W_)
                    else:  # dis
                        c = h * feature_W_ + w
                        grad_mask_com = grad_output[n, c, :, :].view(feature_H_, feature_W_)
                    grad_mask_ori = torch.zeros(mask_H_, mask_W_, dtype=grad_output.dtype, device=grad_output.device)
                    grad_mask_ori[hstart:hend, wstart:wend] = grad_mask_com[hstart_com:hend_com, wstart_com:wend_com]
                    grad_input[n, :, h, w] = grad_mask_ori.view(-1)
        '''
        # c/cuda implementation
        if not grad_output.is_cuda:
            src.cpu.psamask_backward(self.psa_type, grad_output, grad_input, num_, feature_H_, feature_W_, mask_H_, mask_W_, half_mask_H_, half_mask_W_)
        else:
            src.gpu.psamask_backward(self.psa_type, grad_output, grad_input, num_, feature_H_, feature_W_, mask_H_, mask_W_, half_mask_H_, half_mask_W_)
        return grad_input
