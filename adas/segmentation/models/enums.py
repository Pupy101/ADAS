from enum import Enum


class ModelSize(Enum):
    """Model sizes"""

    SMALL = "SMALL"
    MEDIUM = "MEDIUM"
    BIG = "BIG"


class ModelType(Enum):
    """Models types"""

    UNET = "UNET"
    U2NET = "U2NET"
