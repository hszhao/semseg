import torch
from torch.autograd import Function
from .. import src


class PSAMask(Function):
    @staticmethod
    def forward(ctx, input, psa_type=0, mask_H_=None, mask_W_=None):
        assert psa_type in [0, 1]  # 0-col, 1-dis
        assert (mask_H_ is None and mask_W_ is None) or (mask_H_ is not None and mask_W_ is not None)
        num_, channels_, feature_H_, feature_W_ = input.size()
        if mask_H_ is None and mask_W_ is None:
            mask_H_, mask_W_ = 2 * feature_H_ - 1, 2 * feature_W_ - 1
        assert (mask_H_ % 2 == 1) and (mask_W_ % 2 == 1)
        assert channels_ == mask_H_ * mask_W_
        half_mask_H_, half_mask_W_ = (mask_H_ - 1) // 2, (mask_W_ - 1) // 2
        output = torch.zeros([num_, feature_H_ * feature_W_, feature_H_, feature_W_], dtype=input.dtype, device=input.device)
        if not input.is_cuda:
            src.cpu.psamask_forward(psa_type, input, output, num_, feature_H_, feature_W_, mask_H_, mask_W_, half_mask_H_, half_mask_W_)
        else:
            output = output.cuda()
            src.gpu.psamask_forward(psa_type, input, output, num_, feature_H_, feature_W_, mask_H_, mask_W_, half_mask_H_, half_mask_W_)
        ctx.psa_type, ctx.num_, ctx.channels_, ctx.feature_H_, ctx.feature_W_ = psa_type, num_, channels_, feature_H_, feature_W_
        ctx.mask_H_, ctx.mask_W_, ctx.half_mask_H_, ctx.half_mask_W_ = mask_H_, mask_W_, half_mask_H_, half_mask_W_
        return output

    @staticmethod
    def backward(ctx, grad_output):
        psa_type, num_, channels_, feature_H_, feature_W_ = ctx.psa_type, ctx.num_, ctx.channels_, ctx.feature_H_, ctx.feature_W_
        mask_H_, mask_W_, half_mask_H_, half_mask_W_ = ctx.mask_H_, ctx.mask_W_, ctx.half_mask_H_, ctx.half_mask_W_
        grad_input = torch.zeros([num_, channels_, feature_H_, feature_W_], dtype=grad_output.dtype, device=grad_output.device)
        if not grad_output.is_cuda:
            src.cpu.psamask_backward(psa_type, grad_output, grad_input, num_, feature_H_, feature_W_, mask_H_, mask_W_, half_mask_H_, half_mask_W_)
        else:
            src.gpu.psamask_backward(psa_type, grad_output, grad_input, num_, feature_H_, feature_W_, mask_H_, mask_W_, half_mask_H_, half_mask_W_)
        return grad_input, None, None, None


psa_mask = PSAMask.apply
