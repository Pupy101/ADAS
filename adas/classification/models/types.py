from enum import Enum

from adas.core.models.types import DownsampleMode  # pylint: disable=unused-import
from adas.segmentation.models.types import ModelSize  # pylint: disable=unused-import


class ModelType(Enum):
    UNET_ENCODER = "Unet_encoder"
    U2NET_ENCODER = "U2net_encoder"
