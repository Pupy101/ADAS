from typing import Callable, List, Optional, TypeVar

import albumentations as A
from albumentations.pytorch import ToTensorV2

from .types import DatasetType

Transform = TypeVar("Transform", bound=A.BasicTransform)


def create_image_augmentation(
    dataset_type: str,
    crop_transforms: Optional[List[Transform]] = None,
    augmentation_transforms: Optional[List[Transform]] = None,
    normalize_transform: Optional[List[Transform]] = None,
    to_tensor_transform: Optional[List[Transform]] = None,
) -> Callable:
    """Creation augmentation for train or validation"""
    if dataset_type == DatasetType.TRAIN.value:
        crop_transforms = crop_transforms or [
            A.SmallestMaxSize(max_size=280),
            A.RandomCrop(height=224, width=224, always_apply=True),
        ]
        augmentation_transforms = augmentation_transforms or [
            A.Blur(p=0.3),
            A.CoarseDropout(max_height=16, max_width=16, p=0.3),
            A.HorizontalFlip(p=0.5),
            A.HueSaturationValue(p=0.3),
            A.RandomBrightnessContrast(p=0.3),
            A.RGBShift(r_shift_limit=30, g_shift_limit=30, b_shift_limit=30, p=0.3),
        ]
    elif dataset_type in [DatasetType.VALID.value, DatasetType.TEST.value]:
        crop_transforms = crop_transforms or [
            A.SmallestMaxSize(max_size=280),
            A.CenterCrop(height=224, width=224, always_apply=True),
        ]
        augmentation_transforms = augmentation_transforms or []
    else:
        acceptable_types = [_.value for _ in DatasetType]
        raise ValueError(
            f"Strange dataset_type: '{dataset_type}'. Acceptable types: {acceptable_types}"
        )
    normalize_transform = normalize_transform or [A.Normalize()]
    to_tensor_transform = to_tensor_transform or [ToTensorV2()]
    return A.Compose(
        crop_transforms + augmentation_transforms + normalize_transform + to_tensor_transform
    )
