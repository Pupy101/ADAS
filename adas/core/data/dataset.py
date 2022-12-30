from pathlib import Path
from typing import List, Set, Union

import numpy as np
from PIL import Image


class MixingDataset:
    """Mixing class for dataset"""

    @staticmethod
    def open_image(path: Union[str, Path]) -> np.ndarray:
        """Open image with PIL"""
        return np.array(Image.open(path))

    @staticmethod
    def find_files(directory: Union[str, Path], extensions: Set[str]) -> List[Path]:
        """Find all files with setted extension"""
        return [file for file in Path(directory).rglob("*") if file.suffix.lower() in extensions]
