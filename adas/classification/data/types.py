from dataclasses import dataclass
from pathlib import Path


@dataclass
class ImageAndLabel:
    """Pair of image and it's label"""

    image: Path
    label: int
