from typing import List, Tuple, Union

import torch
from torch import Tensor, nn

from .blocks import ModuleWithDevice
from .configs import (
    DownsampleX2BlockConfig,
    DWConv2dBNLReLUConfig,
    DWConv2dConfig,
    ManyConfigs,
    UpsamplePredictHeadConfig,
    UpsampleX2BlockConfig,
)
from .enums import ModelSize


class UnetEncoder(ModuleWithDevice):
    """Unet decoder"""

    def __init__(self, in_channels: int, size: ModelSize, max_pool: bool) -> None:
        super().__init__()
        self.size = size
        shapes: Tuple[int, int, int, int]
        if size is ModelSize.BIG:
            shapes = (64, 128, 256, 512)
        elif size is ModelSize.MEDIUM:
            shapes = (32, 64, 128, 256)
        elif size is ModelSize.SMALL:
            shapes = (16, 32, 64, 128)
        else:
            raise ValueError(f"Strange size for UnetEncoder: {size}")
        configuration = [
            ManyConfigs(
                [
                    DWConv2dBNLReLUConfig(in_channels=in_channels, out_channels=shapes[0]),
                    DWConv2dBNLReLUConfig(in_channels=shapes[0], out_channels=shapes[0]),
                ]
            ),
            ManyConfigs(
                [
                    DWConv2dBNLReLUConfig(in_channels=shapes[0], out_channels=shapes[1]),
                    DWConv2dBNLReLUConfig(in_channels=shapes[1], out_channels=shapes[1]),
                ]
            ),
            ManyConfigs(
                [
                    DWConv2dBNLReLUConfig(in_channels=shapes[1], out_channels=shapes[2]),
                    DWConv2dBNLReLUConfig(in_channels=shapes[2], out_channels=shapes[2]),
                ]
            ),
            ManyConfigs(
                [
                    DWConv2dBNLReLUConfig(in_channels=shapes[2], out_channels=shapes[3]),
                    DWConv2dBNLReLUConfig(in_channels=shapes[3], out_channels=shapes[3]),
                ]
            ),
        ]
        downsample_configuration: List[DownsampleX2BlockConfig] = [
            DownsampleX2BlockConfig(in_channels=c.configs[-1].out_channels, max_pool=max_pool)
            for c in configuration
        ]
        self.encoder_stages = nn.ModuleList([c.create() for c in configuration])
        self.downsample_stages = nn.ModuleList([c.create() for c in downsample_configuration])

    def forward(self, batch: Tensor) -> Tuple[Tensor, List[Tensor]]:
        """Forward step of module"""
        encoder_outputs = []
        for encoder_stage, down_stage in zip(self.encoder_stages, self.downsample_stages):
            batch = encoder_stage(batch)
            encoder_outputs.append(batch)
            batch = down_stage(batch)
        return batch, encoder_outputs


class UnetBottleneck(ModuleWithDevice):
    """U2net bottleneck"""

    def __init__(self, size: ModelSize) -> None:
        super().__init__()
        shapes: Tuple[int, int, int]
        if size is ModelSize.BIG:
            shapes = (512, 1024, 512)
        elif size is ModelSize.MEDIUM:
            shapes = (256, 512, 256)
        elif size is ModelSize.SMALL:
            shapes = (128, 256, 128)
        else:
            raise ValueError(f"Strange size for UnetBottleneck: {size}")
        configuration = ManyConfigs(
            [
                DWConv2dBNLReLUConfig(in_channels=shapes[0], out_channels=shapes[1]),
                DWConv2dBNLReLUConfig(in_channels=shapes[1], out_channels=shapes[2]),
            ]
        )
        self.bottleneck = configuration.create()

    def forward(self, batch: Tensor) -> Tensor:
        """Forward step of module"""
        return self.bottleneck(batch)


class UnetDecoder(ModuleWithDevice):
    """Unet decoder"""

    def __init__(self, size: ModelSize, bilinear: bool) -> None:
        super().__init__()
        shapes: Tuple[int, int, int, int, int]
        if size is ModelSize.BIG:
            shapes = (1024, 512, 256, 128, 64)
        elif size is ModelSize.MEDIUM:
            shapes = (512, 256, 128, 64, 32)
        elif size is ModelSize.SMALL:
            shapes = (256, 128, 64, 32, 16)
        else:
            raise ValueError(f"Strange size for UnetDecoder: {size}")
        configuration = [
            ManyConfigs(
                [
                    DWConv2dBNLReLUConfig(in_channels=shapes[0], out_channels=shapes[1]),
                    DWConv2dBNLReLUConfig(in_channels=shapes[1], out_channels=shapes[2]),
                ]
            ),
            ManyConfigs(
                [
                    DWConv2dBNLReLUConfig(in_channels=shapes[1], out_channels=shapes[2]),
                    DWConv2dBNLReLUConfig(in_channels=shapes[2], out_channels=shapes[3]),
                ]
            ),
            ManyConfigs(
                [
                    DWConv2dBNLReLUConfig(in_channels=shapes[2], out_channels=shapes[3]),
                    DWConv2dBNLReLUConfig(in_channels=shapes[3], out_channels=shapes[4]),
                ]
            ),
            ManyConfigs(
                [
                    DWConv2dBNLReLUConfig(in_channels=shapes[3], out_channels=shapes[4]),
                    DWConv2dBNLReLUConfig(in_channels=shapes[4], out_channels=shapes[4]),
                ]
            ),
        ]
        upsample_configuration: List[UpsampleX2BlockConfig] = [
            UpsampleX2BlockConfig(in_channels=s, bilinear=bilinear) for s in shapes[1:]
        ]
        self.upsample_stages = nn.ModuleList([c.create() for c in upsample_configuration])
        self.decoder_stages = nn.ModuleList([c.create() for c in configuration])

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

    def __init__(self, out_channels: int, size: ModelSize, count_predict_masks: int = 5) -> None:
        super().__init__()
        assert 0 < count_predict_masks < 5, "Count features from encoder maximum is 4"
        self.count_predict_masks = count_predict_masks
        in_channels: Tuple[int, int, int, int]
        if size is ModelSize.BIG:
            in_channels = (256, 128, 64, 64)
        elif size is ModelSize.MEDIUM:
            in_channels = (128, 64, 32, 32)
        elif size is ModelSize.SMALL:
            in_channels = (64, 32, 16, 16)
        else:
            raise ValueError(f"Strange size for U2netPredictHeads: {size}")
        configuration: List[Union[UpsamplePredictHeadConfig, DWConv2dConfig]] = [
            UpsamplePredictHeadConfig(
                scale_factor=8,
                in_channels=in_channels[-4],
                out_channels=out_channels,
            ),
            UpsamplePredictHeadConfig(
                scale_factor=4,
                in_channels=in_channels[-3],
                out_channels=out_channels,
            ),
            UpsamplePredictHeadConfig(
                scale_factor=2,
                in_channels=in_channels[-2],
                out_channels=out_channels,
            ),
            DWConv2dConfig(in_channels=in_channels[-1], out_channels=out_channels),
        ]
        self.heads = nn.ModuleList([c.create() for c in configuration])

    def forward(self, decoder_outputs: List[Tensor]) -> Tuple[Tensor, ...]:
        """Forward step of module"""
        headers_outputs = []
        for decoder_batch, clf_head in zip(
            decoder_outputs[-self.count_predict_masks :], self.heads[-self.count_predict_masks :]
        ):
            headers_outputs.append(clf_head(decoder_batch))
        return tuple(headers_outputs)


class Unet(ModuleWithDevice):
    """Unet model https://arxiv.org/pdf/1505.04597.pdf"""

    def __init__(  # pylint: disable=too-many-arguments
        self,
        in_channels: int,
        out_channels: int,
        size: ModelSize,
        max_pool: bool,
        bilinear: bool,
        count_predict_masks: int,
    ) -> None:
        """Model init"""
        super().__init__()

        self.encoder = UnetEncoder(in_channels=in_channels, size=size, max_pool=max_pool)
        self.bottleneck = UnetBottleneck(size=size)
        self.decoder = UnetDecoder(size=size, bilinear=bilinear)
        self.heads = UnetPredictHeads(
            out_channels=out_channels, size=size, count_predict_masks=count_predict_masks
        )

    def forward(self, batch: Tensor) -> Tuple[Tensor, ...]:
        """Forward step of Unet"""
        batch, encoder_outputs = self.encoder.forward(batch=batch)
        batch = self.bottleneck.forward(batch=batch)
        decoder_outputs = self.decoder.forward(batch=batch, encoder_outputs=encoder_outputs)
        return self.heads.forward(decoder_outputs=decoder_outputs)
