from __future__ import annotations

import copy
import enum
import logging

import torch
from torch import nn

from faultforge.cifar.dataset import Cifar

logger = logging.getLogger(__name__)

_ROOT_MODULE_CACHE: dict[CifarModel, nn.Module] = dict()


class CifarModel(enum.Enum):
    """A CIFAR model with a global cache."""

    ResNet20 = "resnet20"
    ResNet32 = "resnet32"
    ResNet56 = "resnet56"
    Vgg11 = "vgg11_bn"
    Vgg13 = "vgg13_bn"
    Vgg16 = "vgg16_bn"
    Vgg19 = "vgg19_bn"
    MobileNetV2_X0_5 = "mobilenetv2_x0_5"
    MobileNetV2_X0_75 = "mobilenetv2_x0_75"
    MobileNetV2_X1_0 = "mobilenetv2_x1_0"
    MobileNetV2_X1_5 = "mobilenetv2_x1_5"
    ShuffleNetV2_X0_5 = "shufflenetv2_x0_5"
    ShuffleNetV2_X1_0 = "shufflenetv2_x1_0"
    ShuffleNetV2_X1_5 = "shufflenetv2_x1_5"
    ShuffleNetV2_X2_0 = "shufflenetv2_x2_0"
    RepVggA0 = "repvgg_a0"
    RepVggA1 = "repvgg_a1"
    RepVggA2 = "repvgg_a2"

    def root_module(self, dataset: Cifar) -> nn.Module:
        model = _ROOT_MODULE_CACHE.get(self)

        if model is None:
            model = torch.hub.load(  # pyright: ignore[reportUnknownMemberType]
                "chenyaofo/pytorch-cifar-models",
                f"{dataset.kind.value}_{self.value}",
                pretrained=True,
            )
            assert isinstance(model, nn.Module)
            _ROOT_MODULE_CACHE[self] = model

        return copy.deepcopy(model)

    @staticmethod
    def clear_cache():
        """Clear the cached models."""
        _ROOT_MODULE_CACHE.clear()
