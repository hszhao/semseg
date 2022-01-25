import math
import warnings
from typing import Callable, Set
from typing import Type

import torch

from acosp.masks.mask_layer import ChannelMaskLayer
from acosp.masks.sigmoid_soft_top_k import SigmoidSoftTopK
from acosp.masks.soft_top_k import SoftTopK
from acosp.masks.sparse_conv import ChannelSparseConv


class MaskingSequential(torch.nn.Sequential):
    """Light weight sequential to put the mask layer after another layer and get flops/mask information."""

    @property
    def __flops__(self):
        """Get flops from first layer."""
        return self[0].__flops__

    @property
    def mask(self) -> ChannelMaskLayer:
        """Get mask from second layer."""
        return self[1].mask


class UnprunableConv2d(torch.nn.Conv2d):
    """Dummy conv class that can be used to flag individual layers as unprunable."""

    pass


def apply_to_modules(
    model: torch.nn.Module,
    fn: Callable[[torch.nn.Module, str], torch.nn.Module],
    target_classes: Set[Type[torch.nn.Module]],
    forbidden_classes: Set[Type[torch.nn.Module]] = None,
    prefix: str = "",
) -> None:
    """Replace all modules whose type is in target_classes and not in the forbidden classes with result of given
    function.

    Args:
        model: Model
        target_classes: Targeted types.
        forbidden_classes: Types to be untouched.
        fn: Function to be applied to targeted modules.
        prefix: Prefix used to track nested name of module. Useful for logging.
    """

    for module_name, module in model.named_children():
        prefix = f"{prefix}{module_name}/"

        # nested module ==> go inside it
        apply_to_modules(module, fn, target_classes, forbidden_classes, prefix=prefix)

        instance_of_targets = any(isinstance(module, tp) for tp in target_classes)
        instance_of_forbidden = forbidden_classes is not None and any(
            isinstance(module, tp) for tp in forbidden_classes
        )
        if instance_of_targets and not instance_of_forbidden:
            # Replace module with result of function
            setattr(model, module_name, fn(module, prefix))


def add_soft_k_to_model(
    model: torch.nn.Module,
    min_remaining_elements: int,
    selection_factor: float,
    target_classes: Set[Type[torch.nn.Module]],
    forbidden_classes: Set[Type[torch.nn.Module]],
    soft_k_class: Type[SoftTopK] = SigmoidSoftTopK,
    **kwargs,
) -> None:
    """Add soft k masks to the given model."""

    def mask_initializer(module: torch.nn.Module, prefix: str) -> MaskingSequential:
        k = max(
            math.ceil(selection_factor * module.out_channels),
            min_remaining_elements,
        )
        mask = soft_k_class(
            k=k,
            num_channels=module.out_channels,
            **kwargs,
        )

        return MaskingSequential(module, mask)

    apply_to_modules(
        model,
        fn=mask_initializer,
        target_classes=target_classes,
        forbidden_classes=forbidden_classes,
    )


def soft_to_hard_k(model: torch.nn.Module) -> None:
    """Convert all soft k masks in the given model to binary masks."""

    def mask_initializer(module: SoftTopK, _: str) -> torch.nn.Module:
        soft_mask = module.mask.detach()
        q = 1 - module.k / len(soft_mask)
        threshold = torch.quantile(soft_mask, q).detach().item()

        mask = (soft_mask >= threshold).float().detach()
        print(f"Creating hard mask {soft_mask}->{mask}")

        mask_layer = ChannelMaskLayer(mask)

        if mask.sum() != module.k:
            warnings.warn(
                "Incorrect number of channels are masked: {mask} for {module.k} number of remaining elements."
            )

        return mask_layer

    apply_to_modules(model, fn=mask_initializer, target_classes={SoftTopK})


def hard_to_conv(model: torch.nn.Module, reinitialize: bool = False) -> None:
    """Convert all hard/soft k masks in the given model to normal convs."""

    def mask_initializer(module: MaskingSequential, _: str) -> torch.nn.Module:
        conv = module[0]
        mask_layer: ChannelMaskLayer = module[1]

        mask = mask_layer.mask.detach()
        conv.weight.requires_grad = False

        # print(conv.weight.shape)

        conv.weight *= mask[:, None, None, None]
        if conv.bias is not None:
            conv.bias.requires_grad = False
            conv.bias *= mask

        if reinitialize:
            weight = torch.zeros_like(conv.weight)
            torch.nn.init.xavier_normal_(weight)
            conv.weight += (1 - mask[:, None, None, None]) * weight

        return conv

    apply_to_modules(model, fn=mask_initializer, target_classes={MaskingSequential})


def soft_to_channel_sparse(model: torch.nn.Module) -> None:
    """Convert all soft k masks in the given model to sparse convs."""

    def mask_initializer(module: MaskingSequential, _: str) -> torch.nn.Module:
        conv = module[0]
        mask_layer = module[1]

        if isinstance(mask_layer, SoftTopK):
            soft_mask = mask_layer.mask.detach()
            q = 1 - mask_layer.k / len(soft_mask)
            threshold = torch.quantile(soft_mask, q).detach().item()

            mask = (soft_mask >= threshold).float().detach()
            print(f"Creating hard mask {soft_mask}->{mask}")
            if mask.sum() != mask_layer.k:
                warnings.warn(
                    "Incorrect number of channels are masked: {mask} for {module.k} number of remaining elements."
                )
        else:
            mask = mask_layer.mask.detach()

        mask_layer = ChannelSparseConv.from_mask(conv, mask)

        return mask_layer

    apply_to_modules(model, fn=mask_initializer, target_classes={MaskingSequential})
