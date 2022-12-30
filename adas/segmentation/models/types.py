from enum import Enum


class ModelSize(Enum):
    """Model sizes"""

    SMALL = "small"
    MEDIUM = "medium"
    BIG = "big"


class ModelType(Enum):
    """Models types"""

    UNET = "Unet"
    U2NET = "U2net"
