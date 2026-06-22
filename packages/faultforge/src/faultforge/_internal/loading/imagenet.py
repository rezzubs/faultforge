import copy
import enum
import logging
import typing
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import override

import timm
import torchvision
from PIL import Image
from torch import (
    Tensor,
    nn,
)
from torch.utils.data import Dataset
from torchvision import datasets, transforms

from faultforge._internal.common import AnyPath, DeviceLike
from faultforge._internal.dataset import BatchedDataset
from faultforge._internal.fingerprint import Fingerprint
from faultforge._internal.loading.abc import ModelBundle

logger = logging.getLogger(__name__)

type Transform = Callable[[Image.Image], Tensor]


def _get_tim_transform(model: nn.Module) -> Transform:
    """Get a Transform for a Module loaded from timm."""
    from timm.data.config import (
        resolve_data_config,
    )
    from timm.data.transforms_factory import create_transform

    pretrained_cfg = model.pretrained_cfg
    config = resolve_data_config(pretrained_cfg=pretrained_cfg)
    assert isinstance(config, object)

    if not isinstance(config, dict):
        raise TypeError(f"Expected config to be a dict, got {type(config)}")

    transform = create_transform(**config)

    if not isinstance(transform, transforms.Compose):
        raise TypeError(f"Expected transform to be a Compose, got {type(transform)}")

    return typing.cast(Transform, transform)


class ImageNetModel(enum.Enum):
    # Hugging Face models
    DeitTiny = "deit_tiny_patch16_224"
    DeitBase = "deit_base_patch16_224"
    SwinTiny = "swin_tiny_patch4_window7_224"
    VitBase = "vit_base_patch16_224"
    VitTiny = "vit_tiny_patch16_224"

    # Torchvision models
    InceptionV3 = "inception_v3"
    MobileNetV2 = "mobilenet_v2"
    ResNet152 = "resnet152"


@dataclass(slots=True)
class ImageNet(ModelBundle):
    """A Description for loading the Imagenet dataset and models.

    The model is cached inside the instance on first load and `load_model`
    returns a copies of the cached model.
    """

    _root: AnyPath
    _kind: ImageNetModel
    _model: nn.Module | None
    """A cached copy of the loaded model.

    Needs to be cached to load the dataset. See _get_tim_transform.
    """

    def __init__(self, kind: ImageNetModel, root: AnyPath):
        self._root = root
        self._kind = kind
        self._model = None

    def _load_model(self) -> nn.Module:
        logger.info(f"Loading model {self._kind.name} for dataset ImageNet.")
        match self._kind:
            case (
                ImageNetModel.DeitTiny
                | ImageNetModel.DeitBase
                | ImageNetModel.SwinTiny
                | ImageNetModel.VitBase
                | ImageNetModel.VitTiny
            ):
                root_module = timm.create_model(self._kind.value, pretrained=True)
            case ImageNetModel.InceptionV3:
                root_module = torchvision.models.inception_v3(
                    weights=torchvision.models.Inception_V3_Weights.IMAGENET1K_V1
                )
            case ImageNetModel.MobileNetV2:
                root_module = torchvision.models.mobilenet_v2(
                    weights=torchvision.models.MobileNet_V2_Weights.IMAGENET1K_V2
                )
            case ImageNetModel.ResNet152:
                root_module = torchvision.models.resnet152(
                    weights=torchvision.models.ResNet152_Weights.IMAGENET1K_V2
                )

        logger.debug(f"Done loading model {self._kind.name}.")

        return root_module

    def _cached_model(self) -> nn.Module:
        if self._model is None:
            self._model = self._load_model()
        return self._model

    def get_transform(self) -> Transform:
        """Get the proper preprocessing transform for this model."""

        match self._kind:
            case (
                ImageNetModel.DeitTiny
                | ImageNetModel.DeitBase
                | ImageNetModel.SwinTiny
                | ImageNetModel.VitBase
                | ImageNetModel.VitTiny
            ):
                return _get_tim_transform(self._cached_model())
            case ImageNetModel.InceptionV3:
                weights = torchvision.models.Inception_V3_Weights.IMAGENET1K_V1
            case ImageNetModel.MobileNetV2:
                weights = torchvision.models.MobileNet_V2_Weights.IMAGENET1K_V2
            case ImageNetModel.ResNet152:
                weights = torchvision.models.ResNet152_Weights.IMAGENET1K_V2

        return weights.transforms()

    @override
    def fingerprint(self) -> Fingerprint:
        return Fingerprint(kind="imagenet", scalars={"model": self._kind.value})

    @override
    def load_model(self, device: DeviceLike) -> nn.Module:
        return copy.deepcopy(self._cached_model()).to(device)

    @override
    def load_dataset(self, batch_size: int, device: DeviceLike) -> BatchedDataset:
        logger.info("Loading ImageNet.")
        dataset = datasets.ImageNet(
            Path(self._root), split="val", transform=self.get_transform()
        )
        assert isinstance(dataset, Dataset)
        logger.debug("Done loading ImageNet")
        return BatchedDataset.from_dataset(dataset, batch_size, device)
