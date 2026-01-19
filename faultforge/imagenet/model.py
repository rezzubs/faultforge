"""ImageNet DNN models."""

import copy
import enum
import logging
from typing import cast

import timm
import torchvision
from torch import nn
from torchvision import transforms

from faultforge.imagenet.dataset import Transform

_logger = logging.getLogger(__name__)

_root_module_cache: dict[Model, nn.Module] = dict()


def _get_tim_transform(model: nn.Module) -> Transform:
    from timm.data.config import (
        resolve_data_config,  # pyright: ignore[reportUnknownVariableType]
    )
    from timm.data.transforms_factory import create_transform

    pretrained_cfg = model.pretrained_cfg
    config = resolve_data_config(pretrained_cfg=pretrained_cfg)  # pyright: ignore[reportUnknownVariableType]
    assert isinstance(config, object)

    if not isinstance(config, dict):
        raise TypeError(f"Expected config to be a dict, got {type(config)}")

    transform = create_transform(**config)  # pyright: ignore[reportUnknownArgumentType]

    if not isinstance(transform, transforms.Compose):
        raise TypeError(f"Expected transform to be a Compose, got {type(transform)}")

    return cast(Transform, transform)


class Model(enum.Enum):
    """ImageNet DNN models."""

    # Hugging Face models
    DeitTiny = "deit_tiny_patch16_224"
    DeitBase = "deit_base_patch16_224"
    SwinTiny = "swin_tiny_patch4_window7_224"
    VitBase = "vit_base_patch16_224"
    VitTiny = "vit_tiny_patch16_224"

    # Torchvision models
    InceptionV3 = "inception_v3"
    MobileNetV2 = "mobilenet_v2"
    Resnet152 = "resnet152"

    def _get_root_module(self) -> nn.Module:
        root_module = _root_module_cache.get(self, None)

        if root_module is not None:
            _logger.debug("Using cached root module")
            return root_module

        _logger.debug("Loading root module.")

        match self:
            case (
                Model.DeitTiny
                | Model.DeitBase
                | Model.SwinTiny
                | Model.VitBase
                | Model.VitTiny
            ):
                root_module = timm.create_model(self.value, pretrained=True)
            case Model.InceptionV3:
                root_module = torchvision.models.inception_v3(
                    weights=torchvision.models.Inception_V3_Weights.IMAGENET1K_V1
                )
            case Model.MobileNetV2:
                root_module = torchvision.models.mobilenet_v2(
                    weights=torchvision.models.MobileNet_V2_Weights.IMAGENET1K_V2
                )
            case Model.Resnet152:
                root_module = torchvision.models.resnet152(
                    weights=torchvision.models.ResNet152_Weights.IMAGENET1K_V2
                )

        _root_module_cache[self] = root_module

        _logger.debug("Done loading root module.")

        return root_module

    def get_root_module(self) -> nn.Module:
        """Get a copy of a cached root module."""
        return copy.deepcopy(self._get_root_module())

    def get_transform(self) -> Transform:
        """Get the proper preprocessing transform for this model."""

        match self:
            case (
                Model.DeitTiny
                | Model.DeitBase
                | Model.SwinTiny
                | Model.VitBase
                | Model.VitTiny
            ):
                return _get_tim_transform(self._get_root_module())
            case Model.InceptionV3:
                weights = torchvision.models.Inception_V3_Weights.IMAGENET1K_V1
            case Model.MobileNetV2:
                weights = torchvision.models.MobileNet_V2_Weights.IMAGENET1K_V2
            case Model.Resnet152:
                weights = torchvision.models.ResNet152_Weights.IMAGENET1K_V2

        return weights.transforms()  # pyright: ignore[reportAny]
