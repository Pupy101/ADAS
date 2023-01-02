from pathlib import Path
from typing import Callable, Dict, List, Optional, Set, Union

import pandas as pd
from torch import Tensor
from torch.utils.data import Dataset

from adas.core.data.dataset import DatasetMixin

from .types import ImageAndLabel


class ImageClassificationDataset(Dataset, DatasetMixin):
    """Dataset for train image classification"""

    EXTENSIONS: Set[str] = {".png", ".jpg", ".jpeg"}

    def __init__(self, data: List[ImageAndLabel], transforms: Callable) -> None:
        self.data = data
        self.transforms = transforms

    def __getitem__(self, ind: int) -> Dict[str, Union[int, Tensor]]:
        pair = self.data[ind]
        data = self.transforms(image=self.open_image(pair.image))
        image: Tensor = data["image"]
        return {"features": image, "targets": pair.label}

    def __len__(self) -> int:
        return len(self.data)

    @staticmethod
    def found_dataset_data_from_dir(
        data_dir: Union[str, Path], extensions: Optional[Set[str]] = None
    ) -> List[ImageAndLabel]:
        """
        Found dataset data from directory with structure:
        dataset_dir/
            label_1/
                file_1
                file_2
                ...
            label_2/
                file_1
                file_2
                ...
            ...
        """
        extensions = extensions or ImageClassificationDataset.EXTENSIONS
        files = ImageClassificationDataset.find_files(directory=data_dir, extensions=extensions)
        directories = sorted({_.parent.name for _ in files})
        dir2label = {_: i for i, _ in enumerate(directories)}
        data: List[ImageAndLabel] = []
        for file in files:
            dir_name = file.parent.name
            if dir_name in dir2label:
                data.append(ImageAndLabel(image=file, label=dir2label[dir_name]))
        return data

    @staticmethod
    def found_dataset_data_from_dataframe(
        dataframe: pd.DataFrame, image_col: str = "image", label_col: str = "label"
    ) -> List[ImageAndLabel]:
        """Found dataset data from pandas.DataFrame"""
        data: List[ImageAndLabel] = []
        for _, row in dataframe.iterrows():
            image = Path(row[image_col])
            label = int(row[label_col])
            if not image.exists():
                continue
            data.append(ImageAndLabel(image=image, label=label))
        return data
