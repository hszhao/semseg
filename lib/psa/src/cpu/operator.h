#include <torch/torch.h>

void psamask_forward_cpu(const int psa_type, const at::Tensor& input, at::Tensor& output, const int num_, const int feature_H_, const int feature_W_, const int mask_H_, const int mask_W_, const int half_mask_H_, const int half_mask_W_);
void psamask_backward_cpu(const int psa_type, const at::Tensor& grad_output, at::Tensor& grad_input, const int num_, const int feature_H_, const int feature_W_, const int mask_H_, const int mask_W_, const int half_mask_H_, const int half_mask_W_);