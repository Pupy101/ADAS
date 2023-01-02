from torch import Tensor, nn

from adas.core.models.blocks import DWConv2dBNLReLU
from adas.segmentation.models.u2net import U2netEncoder
from adas.segmentation.models.unet import UnetEncoder

from .types import ModelSize


class UnetEncoderClassifier(UnetEncoder):
    """Classifier with Unet encoder as backbone"""

    def __init__(
        self, in_channels: int, size: str, downsample_mode: str, count_classes: int
    ) -> None:
        super().__init__(in_channels=in_channels, size=size, downsample_mode=downsample_mode)
        channels: int
        if size == ModelSize.BIG.value:
            channels = 512
        elif size is ModelSize.MEDIUM.value:
            channels = 256
        elif size is ModelSize.SMALL.value:
            channels = 128
        else:
            raise ValueError(f"Strange size for UnetEncoderClassifier: {size}")
        # input to classifier head is [batch_size, <channels>, 14, 14]
        self.classifier = nn.Sequential(
            DWConv2dBNLReLU(in_channels=channels, out_channels=channels * 2, stride=2, padding=0),
            nn.MaxPool2d(kernel_size=2, stride=2),
            DWConv2dBNLReLU(in_channels=channels * 2, out_channels=channels * 3, padding=0),
            nn.Flatten(),
            nn.Linear(in_features=channels * 3, out_features=count_classes),
        )

    def forward(self, batch: Tensor) -> Tensor:  # type: ignore
        output, _ = super().forward(batch)
        return self.classifier(output)


class U2netEncoderClassifier(U2netEncoder):
    """Classifier with U2net encoder as backbone"""

    def __init__(
        self, in_channels: int, size: str, downsample_mode: str, count_classes: int
    ) -> None:
        super().__init__(in_channels=in_channels, size=size, downsample_mode=downsample_mode)
        if size == ModelSize.BIG.value:
            channels = 512
        elif size is ModelSize.MEDIUM.value:
            channels = 256
        elif size is ModelSize.SMALL.value:
            channels = 128
        else:
            raise ValueError(f"Strange size for UnetEncoderClassifier: {size}")
        self.classifier = nn.Sequential(
            DWConv2dBNLReLU(in_channels=channels, out_channels=channels * 2, stride=2, padding=0),
            DWConv2dBNLReLU(in_channels=channels * 2, out_channels=channels * 3, padding=0),
            nn.Flatten(),
            nn.Linear(in_features=channels * 3, out_features=count_classes),
        )

    def forward(self, batch: Tensor) -> Tensor:  # type: ignore
        output, _ = super().forward(batch)
        return self.classifier(output)
