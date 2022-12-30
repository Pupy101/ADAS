from dataclasses import dataclass
from pathlib import Path


@dataclass
class ImageAndMask:
    """Pair of image and it's mask"""

    image: Path
    mask: Path
