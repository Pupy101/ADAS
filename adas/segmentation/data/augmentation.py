from typing import Callable

import albumentations as A
from albumentations.pytorch import ToTensorV2


def create_train_augmentation(is_train: bool = True) -> Callable:
    if is_train:
        return A.Compose(
            [
                A.SmallestMaxSize(max_size=240),
                A.HorizontalFlip(p=0.5),
                A.RandomCrop(height=224, width=224, always_apply=True),
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.3),
                A.GridDistortion(p=0.3),
                A.HueSaturationValue(p=0.3),
                A.Normalize(),
                ToTensorV2(),
            ]
        )
    return A.Compose(
        [
            A.SmallestMaxSize(max_size=352),
            A.CenterCrop(height=224, width=224, always_apply=True),
            A.Normalize(),
            ToTensorV2(),
        ]
    )
