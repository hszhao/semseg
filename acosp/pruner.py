import functools
from collections import defaultdict
from typing import Callable

import torch

from acosp import inject
from acosp.masks import mask_layer
from acosp.masks.soft_top_k import SoftTopK


def while_pruning(fn: Callable) -> Callable:
    """Decorator to only apply functions if the pruner is active."""

    @functools.wraps(fn)
    def new_fn(self: object, *args, **kwargs) -> Callable:
        if self.active:
            return fn(self, *args, **kwargs)

    return new_fn


class SoftTopKPruner:
    def __init__(
        self,
        final_sparsity: float,
        ending_epoch: int,
        starting_epoch: int = 0,
        active: bool = True,
        min_remaining_elements: int = 8,
        initial_temp: float = 1,
        final_temp: float = 0.001,
        learnable_weights: bool = True,
        *args,
        **kwargs,
    ) -> None:
        """Initialize class to perform layer wise pruning during training.

        Args:
            final_sparsity: Sparsity to reach in every layer at ending_epoch.
            ending_epoch: Final step of pruning.
            starting_epoch: Initial step of pruning.
            active: Whether pruner is active.
            min_remaining_elements: Minimal number of unpruned elements.
            initial_temp: Used to control "softness".
            final_temp:  Used to control "softness".
            learnable_weights: Whether the mask scaling weights are learnable
            *args
            **kwargs
        """
        super().__init__(*args, **kwargs)
        self.learnable_weights = learnable_weights
        self.active = active
        self.initial_sparsity = 0
        self.final_sparsity = final_sparsity
        self.starting_epoch = starting_epoch
        self.ending_epoch = ending_epoch
        self.min_remaining_elements = min_remaining_elements
        self.final_temp = final_temp
        self.initial_temp = initial_temp

    @while_pruning
    def configure_model(self, model: torch.nn.Module, **kwargs) -> None:
        """Add soft k channel masks to model."""
        inject.add_soft_k_to_model(
            model,
            target_classes={torch.nn.Conv2d, torch.nn.ConvTranspose2d},
            forbidden_classes={inject.UnprunableConv2d},
            selection_factor=1 - self.final_sparsity,
            min_step=self.starting_epoch,
            max_step=self.ending_epoch,
            min_remaining_elements=self.min_remaining_elements,
            initial_temperature=self.initial_temp,
            final_temperature=self.final_temp,
            learnable=self.learnable_weights,
            **kwargs,
        )

    @while_pruning
    def on_pruning_end(self, model: torch.nn.Module) -> None:
        """Hard prune model."""
        inject.soft_to_hard_k(model)
        self.active = False

    @while_pruning
    def update_mask_layers(self, model: torch.nn.Module, epoch: int) -> None:
        """Update masking layers in model. E.g. calculate updated temperatures."""
        trainable_mask_layers = (module for module in model.modules() if isinstance(module, SoftTopK))

        for layer in trainable_mask_layers:
            layer.on_epoch_end(epoch)

    def collect_logs(self, logger: object, model: torch.nn.Module, step: int) -> None:
        """Collect logs from masking layers."""
        prefix = "pruning"
        # Get all mask layers
        mask_layers = (
            (name, module) for name, module in model.named_modules() if isinstance(module, mask_layer.ChannelMaskLayer)
        )

        # Collect logs with layer's name, name of the log value and actual value
        # e.g. (layer4.conv1, temperature, 5)
        mask_layer_logs = (
            (layer_name, log_name, log_val)
            for layer_name, layer in mask_layers
            for log_name, log_val in layer.get_logs().items()
        )

        # Use default dict to collect values from all layers and calculate mean / var of them
        all_values = defaultdict(list)

        for layer_name, log_name, log_val in mask_layer_logs:
            logger.log_metrics({f"{prefix}/{layer_name}/{log_name}": log_val}, step=step)
            all_values[log_name].append(log_val)

        # The rest is there to collect global means/variances/... of all logging values.
        metrics = {
            "mean": torch.mean,
            "var": torch.var,
            "min": torch.min,
            "max": torch.max,
        }
        for name, values in all_values.items():
            metric_dict = {
                f"{prefix}/{metric_name}_{name}": metric_fn(torch.tensor(values, dtype=float)).item()
                for metric_name, metric_fn in metrics.items()
            }
            logger.log_metrics(metric_dict, step)
