import abc
import math

import torch

from acosp.masks.mask_layer import ChannelMaskLayer


class SoftTopK(ChannelMaskLayer, abc.ABC):
    def __init__(
        self,
        k: int,
        num_channels: int,
        min_step: int,
        max_step: int,
        initial_temperature: float,
        final_temperature: float,
        learnable: bool = True,
        *args,
        **kwargs,
    ) -> None:
        """Initialize soft top k masking layer.

        Args:
            k: How many elements to keep.
            num_channels: How many channels there are in total.
            min_step: First step to decay temperature.
            max_step: Step at which temperature reaches final value.
            initial_temperature: Initial temperature.
            final_temperature: Final temperature.
            learnable: Whether scaling logits are learnable or randomly initialized.
        """
        self.k = k

        self.max_step = max_step
        self.min_step = min_step
        self.initial_temperature = initial_temperature
        self.final_temperature = final_temperature

        self.current_temperature = initial_temperature

        if learnable:
            # soft top k
            logits = torch.nn.Parameter(torch.zeros((num_channels,)), requires_grad=True)
        else:
            logits = torch.tensor(torch.zeros((num_channels,)))
        torch.nn.init.normal_(logits)

        super().__init__(mask=torch.ones_like(logits), *args, **kwargs)

        if learnable:
            self.register_parameter("weight", logits)
        else:
            self.register_buffer("weight", logits)

    def calc_temperature(
        self,
        current_step: int,
    ) -> float:
        """Calculate temperature according to an exponential decay function."""
        if current_step <= self.min_step:
            return self.initial_temperature
        assert current_step <= self.max_step

        log_t = math.log(self.initial_temperature) - min(
            (current_step - self.min_step) / (self.max_step - self.min_step), 1
        ) * (math.log(self.initial_temperature) - math.log(self.final_temperature))
        return math.exp(log_t)

    def on_epoch_end(self, current_epoch: int) -> None:
        """Update temperature at the end of each epoch."""
        self.current_temperature = self.calc_temperature(current_epoch)

    def get_logs(self) -> dict:
        """Curate logs."""
        return {"temperature": self.current_temperature, **super().get_logs()}

    @abc.abstractmethod
    def update_mask(self, _: torch.Tensor) -> None:
        """Update masking tensor."""
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Calculate forward pass.

        Update mask weights on the fly.
        """
        if self.training:
            self.update_mask(x)
        return super().forward(x)
