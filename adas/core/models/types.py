from enum import Enum


class UpsampleMode(Enum):
    """Upsample mode"""

    BILINEAR = "bilinear"
    TRANSPOSE_CONVOLUTION = "transpose_convolution"


class DownsampleMode(Enum):
    """Downsample mode"""

    MAX_POOL = "max_pool"
    CONVOLUTION = "convolution"
