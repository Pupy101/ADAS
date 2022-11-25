from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Set, Union

import numpy as np
import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset


@dataclass
class ImageAndMask:
    """Pair of image and it's mask"""

    image: Path
    mask: Path


@dataclass
class ImageAndLabel:
    """Pair of image and label"""

    image: Path
    label: int


class MixingDataset:  # pylint: disable=too-few-public-methods
    """Mixing class for dataset"""

    @staticmethod
    def open(path: Union[str, Path]) -> np.ndarray:
        """Open image with PIL"""
        return np.array(Image.open(path))

    @staticmethod
    def _found_stem2path(
        directory: Union[str, Path], extensions: Optional[Set[str]] = None
    ) -> Dict[str, Path]:
        """Create mapping stem of image file and path to it"""
        if extensions is None:
            extensions = {".png", ".jpg", ".jpeg"}
        directory = Path(directory)
        stem2path: Dict[str, Path] = {}
        for file in directory.rglob("*"):
            if file.suffix.lower() in extensions:
                stem2path[file.stem] = file
        return stem2path


class BDD100KDataset(Dataset, MixingDataset):
    """
    Dataset for train segmentation net based on BD100K
    https://bair.berkeley.edu/blog/2018/05/30/bdd/
    """

    def __init__(self, data: List[ImageAndMask], transforms: Callable) -> None:
        """Dataset init"""
        self.data = data
        self.transforms = transforms

    @staticmethod
    def found_dataset_data(
        image_dir: Union[str, Path],
        mask_dir: Union[str, Path],
        extensions: Optional[Set[str]] = None,
    ) -> List[ImageAndMask]:
        """Finding pairs by name and return that pairs from image and mask folder"""
        images = BDD100KDataset._found_stem2path(directory=image_dir, extensions=extensions)
        masks = BDD100KDataset._found_stem2path(directory=mask_dir, extensions=extensions)
        data = [
            ImageAndMask(image=images[key], mask=masks[key])
            for key in sorted(images.keys() & masks.keys())
        ]
        return data

    def __getitem__(self, ind: int) -> Dict[str, Tensor]:
        pair = self.data[ind]

        data = self.transforms(image=self.open(pair.image), mask=self.open(pair.mask))
        image: Tensor = data["image"]
        mask: Tensor = data["mask"]

        one_hot_mask = torch.zeros(3, mask.size(0), mask.size(1))
        one_hot_mask[0, mask == 0] = 1  # main road
        one_hot_mask[1, mask == 1] = 1  # other roads
        one_hot_mask[2, mask == 2] = 1  # background

        return {"features": image, "targets": one_hot_mask.long()}

    def __len__(self):
        return len(self.data)


class ImageClassificationDataset(Dataset, MixingDataset):
    """Image classification dataset"""

    def __init__(self, data: List[ImageAndLabel], transforms: Callable) -> None:
        """Dataset init"""
        self.data = data
        self.transforms = transforms

    @staticmethod
    def found_dataset_data(
        image_dir: Union[str, Path], extensions: Optional[Set[str]] = None
    ) -> List[ImageAndLabel]:
        """Finding dataset images and it's labels"""
        stem2path = ImageClassificationDataset._found_stem2path(
            directory=image_dir, extensions=extensions
        )
        class_dir2label = {
            _: i for i, _ in enumerate(sorted({p.parent for p in stem2path.values()}))
        }
        data = [
            ImageAndLabel(image=image, label=class_dir2label[image.parent])
            for image in stem2path.values()
        ]
        return data

    def __getitem__(self, ind: int) -> Dict[str, Tensor]:
        pair = self.data[ind]
        image = self.transforms(image=self.open(pair.image))["image"]
        return {"features": image, "targets": torch.tensor(pair.label, dtype=torch.long)}

    def __len__(self) -> int:
        return len(self.data)
