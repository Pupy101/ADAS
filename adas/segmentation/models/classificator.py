from typing import Union

from torch import Tensor, nn

from .blocks import DWConv2dBNLReLU, ModuleWithDevice
from .enums import ModelSize
from .u2net import U2netEncoder
from .unet import UnetEncoder


class Classificator(ModuleWithDevice):
    """Classifiaction head for Unet or U2net encoder"""

    def __init__(
        self,
        feature_extractor: Union[U2netEncoder, UnetEncoder],
        count_classes: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if feature_extractor.size is ModelSize.BIG:
            out_channels = 512
        elif feature_extractor.size is ModelSize.MEDIUM:
            out_channels = 256
        elif feature_extractor.size is ModelSize.SMALL:
            out_channels = 128
        else:
            raise ValueError(f"Strange size for Classificator: {feature_extractor.size}")
        self.feature_extractor = feature_extractor
        self.classificator = nn.Sequential(
            nn.AdaptiveAvgPool2d(3),
            DWConv2dBNLReLU(in_channels=out_channels, out_channels=out_channels, padding=0),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(in_features=out_channels, out_features=count_classes),
        )

    def forward(self, batch: Tensor) -> Tensor:
        """Forward step of Classificator"""
        batch, _ = self.feature_extractor.forward(batch=batch)
        return self.classificator(batch)
