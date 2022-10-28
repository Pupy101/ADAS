from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Set, Tuple, Union

import numpy as np
import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset


@dataclass
class ImageAndMask:
    """Pair of image and its mask"""

    image: Path
    mask: Path


class BDD100KDataset(Dataset):
    """
    Dataset for train segmentation net based on BD100K
    https://bair.berkeley.edu/blog/2018/05/30/bdd/
    """

    def __init__(
        self,
        image_dir: Union[str, Path],
        mask_dir: Union[str, Path],
        transforms: Callable,
        image_extensions: Optional[Set[str]] = None,
    ) -> None:
        """Dataset init"""
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)
        if image_extensions is None:
            image_extensions = {".png", ".jpg"}
        self.img_exts = image_extensions
        self.pairs = self.found_images()
        self.transforms = transforms

    def found_images(self) -> List[ImageAndMask]:
        """Finding pairs by name and return that pairs from image and mask folder"""
        img2path: Dict[str, Path] = {}
        msk2path: Dict[str, Path] = {}
        for file in self.image_dir.rglob("*"):
            if file.suffix in self.img_exts:
                img2path[file.stem] = file
        for file in self.mask_dir.rglob("*"):
            if file.suffix in self.img_exts:
                msk2path[file.stem] = file
        pairs: List[ImageAndMask] = []
        for key in sorted(img2path.keys() & msk2path.keys()):
            pairs.append(ImageAndMask(image=img2path[key], mask=msk2path[key]))
        return pairs

    @staticmethod
    def load_image_and_mask(
        image_and_mask: ImageAndMask,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Open image and mask with PIL"""
        return np.array(Image.open(image_and_mask.image)), np.array(Image.open(image_and_mask.mask))

    def __getitem__(self, ind: int) -> Dict[str, Tensor]:
        img, msk = self.load_image_and_mask(self.pairs[ind])

        data = self.transforms(image=img, mask=msk)
        image: Tensor = data["image"]
        mask: Tensor = data["mask"]

        one_hot_mask = torch.zeros(2, mask.size(0), mask.size(1))
        one_hot_mask[0, mask == 0] = 1  # main road
        one_hot_mask[1, mask == 2] = 1  # backgroud

        return {"features": image, "targets": one_hot_mask.long()}

    def __len__(self):
        return len(self.pairs)
