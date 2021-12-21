from typing import Tuple

import torch
from torch import nn


class ChannelMaskLayer(nn.Module):
    def __init__(self, mask: torch.Tensor) -> None:
        """Initialize masking layer.

        Args:
            mask: Mask.
        """
        super().__init__()
        self.register_buffer("mask", mask)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Calculate masked input."""
        return x * self.mask[None, :, None, None]

    def _calculate_decided_elements(self, eps: float = 0.01) -> Tuple[float, float, float]:
        """Calculate portion of selected / pruned / undecided elements.

        Useful for logging.

        Args:
            eps: Max difference to 0 / 1 to be considered as selected / pruned.

        Returns:
            selected, pruned, undecided elements
        """
        n = len(self.mask)
        chosen = ((self.mask - 1).abs() < eps).sum().item() / n
        pruned = (self.mask.abs() < eps).sum().item() / n
        undecided = 1 - chosen - pruned
        return chosen, pruned, undecided

    def get_logs(self) -> dict:
        """Collect data points that are of interest for logging."""
        chosen, pruned, undecided = self._calculate_decided_elements()

        return {"chosen": chosen, "pruned": pruned, "undecided": undecided}
