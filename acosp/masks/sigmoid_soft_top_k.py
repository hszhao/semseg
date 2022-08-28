from typing import Tuple

import torch

from acosp.masks.soft_top_k import SoftTopK


def sigmoid_soft_top_k(
    logits: torch.Tensor,
    k: int,
    tau: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Calculate soft top k mask using a sigmoid function.

    Args:
        logits: Parameters used for weighting.
        k: Number of elements that should be selected.
        tau: Temperature that is used to vary steepness of sigmoid function.

    Returns:
        mask: Soft top k mask that can be used to select k elements.
        x0: Exactly k (normalized) logits are larger than x0. (Useful to gain insight in logits.)
    """
    if k == len(logits):
        # Not doing anything
        return torch.ones_like(logits), torch.zeros(
            1,
            device=logits.device,
        )

    std, mean = torch.std_mean(logits)
    logits = (logits - mean) / torch.maximum(std, torch.tensor(1e-3))

    # Find splitter value: k values are >=q
    q = 1 - k / len(logits)
    x0 = torch.quantile(logits, q).detach()

    # Center and stretch depending on temperature tau
    logits = (logits - x0) / tau

    return torch.sigmoid(logits), x0.detach()


class SigmoidSoftTopK(SoftTopK):
    def __init__(self, k: int, num_channels: int, *args, **kwargs) -> None:
        """Initialize sigmoid based soft top k masking layer.

        Args:
            k: How many elements to keep.
            num_channels: How many channels there are in total.
        """
        super().__init__(k, num_channels, *args, **kwargs)

        self.x_0 = torch.tensor(0.0)

    def update_mask(self, _: torch.Tensor) -> None:
        """Update masking tensor."""
        self.mask, self.x_0 = sigmoid_soft_top_k(
            self.weight,
            self.k,
            self.current_temperature,
        )

    def get_logs(self) -> dict:
        """Collect data points that are of interest for logging."""
        return {"x_0": self.x_0.item(), **super().get_logs()}

    def eval(self) -> "SoftTopK":
        """Update mask. Not done in forward pass in eval mode."""
        self.mask = sigmoid_soft_top_k(
            self.weight.detach(),
            self.k,
            self.current_temperature,
        )

        return super().eval()
