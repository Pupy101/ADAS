from pathlib import Path
from typing import Callable, Dict, List, Optional, Set, Union

import torch
from torch import Tensor
from torch.utils.data import Dataset

from adas.core.data.dataset import DatasetMixin

from .types import ImageAndMask


class BDD100KDataset(Dataset, DatasetMixin):
    """
    Dataset for train segmentation based on BD100K
    https://bair.berkeley.edu/blog/2018/05/30/bdd/
    """

    EXTENSIONS: Set[str] = {".png", ".jpg", ".jpeg"}

    def __init__(self, data: List[ImageAndMask], transforms: Callable) -> None:
        """Dataset init"""
        self.data = data
        self.transforms = transforms

    def __getitem__(self, ind: int) -> Dict[str, Tensor]:
        pair = self.data[ind]
        data = self.transforms(image=self.open_image(pair.image), mask=self.open_image(pair.mask))
        image: Tensor = data["image"]
        mask: Tensor = data["mask"]

        # transfrom from mask h x w with values [0, 1, 2] into 3 x h x w
        one_hot_mask = torch.zeros(3, mask.size(0), mask.size(1), dtype=torch.long)
        one_hot_mask[0, mask == 0] = 1  # main road
        one_hot_mask[1, mask == 1] = 1  # other roads
        one_hot_mask[2, mask == 2] = 1  # background

        return {"features": image, "targets": one_hot_mask}

    def __len__(self) -> int:
        return len(self.data)

    @staticmethod
    def found_dataset_data(
        image_dir: Union[str, Path],
        mask_dir: Union[str, Path],
        extensions: Optional[Set[str]] = None,
    ) -> List[ImageAndMask]:
        """Found pairs by name and return that pairs from image and mask folder"""
        extensions = extensions or BDD100KDataset.EXTENSIONS
        images = BDD100KDataset.find_files(directory=image_dir, extensions=extensions)
        masks = BDD100KDataset.find_files(directory=mask_dir, extensions=extensions)
        stem2image: Dict[str, Path] = {_.stem: _ for _ in images}
        stem2mask: Dict[str, Path] = {_.stem: _ for _ in masks}
        data = [
            ImageAndMask(image=stem2image[key], mask=stem2mask[key])
            for key in sorted(stem2image.keys() & stem2mask.keys())
        ]
        return data
