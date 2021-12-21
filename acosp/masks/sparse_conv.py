from typing import Sequence

import torch
from torch import nn


def compress_weight(weight: torch.Tensor, indices: Sequence[int]) -> None:
    """Collect weights for given indices."""
    return weight[indices]


class ChannelSparseConv(nn.Module):
    """Very basic and naive convolution layer that supports channel sparse weights.

    Can be used to convert a pruned convolution layer in the form of a conv layer + a mask into a dense conv layer that
    manually sparsifies it's output. In total it saves the unnecessary computations being done when simply masking out
    pruned channels.
    """

    def __init__(
        self,
        indices: Sequence[int],
        conv: torch.nn.Conv2d,
    ) -> None:
        """Convert the convolution into a dense convolution by keeping only the given indices.

        Args:
            indices: List of indices that are not pruned
            conv: Convolution layer.
        """
        assert len(set(indices)) == len(indices), "Indices are not unique."
        assert (
            0 <= min(indices) and max(indices) < conv.out_channels
        ), f"Indices {indices} are out of bounds for conv {conv}."

        super().__init__()

        self.indices = sorted(indices)
        self.out_channels = conv.out_channels

        conv.weight = torch.nn.Parameter(compress_weight(conv.weight, self.indices).detach())
        conv.bias = torch.nn.Parameter(compress_weight(conv.bias, self.indices).detach())

        self.conv = conv

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Calculate channel sparse convolution by only calculating dense channels and inflating the result."""
        input: torch.Tensor = self.conv(input)

        output = torch.zeros(
            (input.shape[0], self.out_channels, *input.shape[2:]),
            device=input.device,
            dtype=input.dtype,
            requires_grad=False,
        )

        output[:, self.indices, ...] += input

        return output

    @classmethod
    def from_mask(cls, conv: nn.Conv2d, mask: torch.Tensor) -> "ChannelSparseConv":
        """Create from conv layer and pruning mask."""
        indices = tuple(torch.where(mask == 1)[0].tolist())
        return cls(indices, conv)
