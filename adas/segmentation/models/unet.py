from typing import List, Tuple

import torch
from torch import Tensor, nn

from adas.core.models.blocks import (
    DownsampleX2Block,
    DWConv2d,
    DWConv2dBNLReLU,
    ModuleWithDevice,
    UpsampleBlock,
    UpsampleX2Block,
)

from .types import ModelSize


class UnetEncoder(ModuleWithDevice):
    """Unet encoder"""

    def __init__(self, in_channels: int, size: str, downsample_mode: str) -> None:
        super().__init__()
        self.size = size
        shapes: Tuple[int, int, int, int]
        if size == ModelSize.BIG.value:
            shapes = (64, 128, 256, 512)
        elif size == ModelSize.MEDIUM.value:
            shapes = (32, 64, 128, 256)
        elif size == ModelSize.SMALL.value:
            shapes = (16, 32, 64, 128)
        else:
            acceptable = [_.value for _ in ModelSize]
            raise ValueError(f"Strange size for UnetEncoder: {size}. Acceptable: {acceptable}")
        self.encoder_stages = nn.ModuleList(
            [
                nn.Sequential(
                    DWConv2dBNLReLU(in_channels=in_channels, out_channels=shapes[0]),
                    DWConv2dBNLReLU(in_channels=shapes[0], out_channels=shapes[0]),
                ),
                nn.Sequential(
                    DWConv2dBNLReLU(in_channels=shapes[0], out_channels=shapes[1]),
                    DWConv2dBNLReLU(in_channels=shapes[1], out_channels=shapes[1]),
                ),
                nn.Sequential(
                    DWConv2dBNLReLU(in_channels=shapes[1], out_channels=shapes[2]),
                    DWConv2dBNLReLU(in_channels=shapes[2], out_channels=shapes[2]),
                ),
                nn.Sequential(
                    DWConv2dBNLReLU(in_channels=shapes[2], out_channels=shapes[3]),
                    DWConv2dBNLReLU(in_channels=shapes[3], out_channels=shapes[3]),
                ),
            ]
        )
        self.downsample_stages = nn.ModuleList(
            [
                DownsampleX2Block(mode=downsample_mode, in_channels=in_channels)
                for in_channels in shapes
            ]
        )

    def forward(self, batch: Tensor) -> Tuple[Tensor, List[Tensor]]:
        """Forward step of UnetEncoder module"""
        encoder_outputs = []
        for encoder_stage, down_stage in zip(self.encoder_stages, self.downsample_stages):
            batch = encoder_stage(batch)
            encoder_outputs.append(batch)
            batch = down_stage(batch)
        return batch, encoder_outputs


class UnetBottleneck(ModuleWithDevice):
    """U2net bottleneck"""

    def __init__(self, size: str) -> None:
        super().__init__()
        shapes: Tuple[int, int, int]
        if size == ModelSize.BIG.value:
            shapes = (512, 1024, 512)
        elif size == ModelSize.MEDIUM.value:
            shapes = (256, 512, 256)
        elif size == ModelSize.SMALL.value:
            shapes = (128, 256, 128)
        else:
            acceptable = [_.value for _ in ModelSize]
            raise ValueError(f"Strange size for UnetBottleneck: {size}. Acceptable: {acceptable}")
        self.bottleneck = nn.Sequential(
            DWConv2dBNLReLU(in_channels=shapes[0], out_channels=shapes[1]),
            DWConv2dBNLReLU(in_channels=shapes[1], out_channels=shapes[2]),
        )

    def forward(self, batch: Tensor) -> Tensor:
        """Forward step of UnetBottleneck module"""
        return self.bottleneck(batch)


class UnetDecoder(ModuleWithDevice):
    """Unet decoder"""

    def __init__(self, size: str, upsample_mode: str) -> None:
        super().__init__()
        shapes: Tuple[int, int, int, int, int]
        if size == ModelSize.BIG.value:
            shapes = (1024, 512, 256, 128, 64)
        elif size == ModelSize.MEDIUM.value:
            shapes = (512, 256, 128, 64, 32)
        elif size == ModelSize.SMALL.value:
            shapes = (256, 128, 64, 32, 16)
        else:
            acceptable = [_.value for _ in ModelSize]
            raise ValueError(f"Strange size for UnetDecoder: {size}. Acceptable: {acceptable}")
        self.decoder_stages = nn.ModuleList(
            [
                nn.Sequential(
                    DWConv2dBNLReLU(in_channels=shapes[0], out_channels=shapes[1]),
                    DWConv2dBNLReLU(in_channels=shapes[1], out_channels=shapes[2]),
                ),
                nn.Sequential(
                    DWConv2dBNLReLU(in_channels=shapes[1], out_channels=shapes[2]),
                    DWConv2dBNLReLU(in_channels=shapes[2], out_channels=shapes[3]),
                ),
                nn.Sequential(
                    DWConv2dBNLReLU(in_channels=shapes[2], out_channels=shapes[3]),
                    DWConv2dBNLReLU(in_channels=shapes[3], out_channels=shapes[4]),
                ),
                nn.Sequential(
                    DWConv2dBNLReLU(in_channels=shapes[3], out_channels=shapes[4]),
                    DWConv2dBNLReLU(in_channels=shapes[4], out_channels=shapes[4]),
                ),
            ]
        )
        self.upsample_stages = nn.ModuleList(
            [
                UpsampleX2Block(mode=upsample_mode, in_channels=in_channels)
                for in_channels in shapes[1:]
            ]
        )

    def forward(self, batch: Tensor, encoder_outputs: List[Tensor]) -> List[Tensor]:
        """Forward step of module"""
        decoder_outputs = []
        for encoder_batch, up_stage, decoder_stage in zip(
            encoder_outputs[::-1], self.upsample_stages, self.decoder_stages
        ):
            batch = decoder_stage(torch.cat([encoder_batch, up_stage(batch)], dim=1))
            decoder_outputs.append(batch)
        return decoder_outputs


class UnetPredictHeads(ModuleWithDevice):
    """Unet predict heads"""

    def __init__(self, out_channels: int, size: str, count_features: int = 4) -> None:
        super().__init__()
        assert 0 < count_features < 5, "Count features from encoder maximum is 4"
        self.count_features = count_features
        shapes: Tuple[int, int, int, int]
        if size == ModelSize.BIG.value:
            shapes = (256, 128, 64, 64)
        elif size == ModelSize.MEDIUM.value:
            shapes = (128, 64, 32, 32)
        elif size == ModelSize.SMALL.value:
            shapes = (64, 32, 16, 16)
        else:
            acceptable = [_.value for _ in ModelSize]
            raise ValueError(f"Strange size for UnetPredictHeads: {size}. Acceptable: {acceptable}")
        heads = [
            nn.Sequential(
                UpsampleBlock(scale_factor=8),
                DWConv2d(in_channels=shapes[0], out_channels=out_channels),
            ),
            nn.Sequential(
                UpsampleBlock(scale_factor=4),
                DWConv2d(in_channels=shapes[1], out_channels=out_channels),
            ),
            nn.Sequential(
                UpsampleBlock(scale_factor=2),
                DWConv2d(in_channels=shapes[2], out_channels=out_channels),
            ),
            DWConv2d(in_channels=shapes[3], out_channels=out_channels),
        ]
        self.heads = nn.ModuleList(heads[-count_features:])

    def forward(self, decoder_outputs: List[Tensor]) -> Tuple[Tensor, ...]:
        """Forward step of module"""
        headers_outputs = []
        for decoder_batch, clf_head in zip(decoder_outputs[-self.count_features :], self.heads):
            headers_outputs.append(clf_head(decoder_batch))
        return tuple(headers_outputs)


class Unet(ModuleWithDevice):
    """Unet model https://arxiv.org/pdf/1505.04597.pdf"""

    def __init__(  # pylint: disable=too-many-arguments
        self,
        in_channels: int,
        out_channels: int,
        size: str,
        downsample_mode: str,
        upsample_mode: str,
        count_features: int,
    ) -> None:
        """Model init"""
        super().__init__()

        self.encoder = UnetEncoder(
            in_channels=in_channels, size=size, downsample_mode=downsample_mode
        )
        self.bottleneck = UnetBottleneck(size=size)
        self.decoder = UnetDecoder(size=size, upsample_mode=upsample_mode)
        self.heads = UnetPredictHeads(
            out_channels=out_channels, size=size, count_features=count_features
        )

    def forward(self, batch: Tensor) -> Tuple[Tensor, ...]:
        """Forward step of Unet"""
        batch, encoder_outputs = self.encoder.forward(batch=batch)
        batch = self.bottleneck.forward(batch=batch)
        decoder_outputs = self.decoder.forward(batch=batch, encoder_outputs=encoder_outputs)
        return self.heads.forward(decoder_outputs=decoder_outputs)
