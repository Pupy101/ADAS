from typing import List, Tuple, Union

import torch
from torch import Tensor, nn

from .blocks import ModuleWithDevice
from .configurations import (
    DownsampleX2BlockConfig,
    DWConv2dConfig,
    ModelSize,
    RSUD1Config,
    RSUD5Config,
    UpsamplePredictHeadConfig,
    UpsampleX2BlockConfig,
)


class U2netEncoder(ModuleWithDevice):
    """U2net encoder"""

    def __init__(self, in_channels: int, size: ModelSize, max_pool: bool) -> None:
        super().__init__()
        self.size = size
        shapes: Tuple[int, int, int, int, int]
        if size is ModelSize.BIG:
            shapes = (32, 64, 128, 256, 512)
        elif size is ModelSize.MEDIUM:
            shapes = (16, 32, 64, 128, 256)
        elif size is ModelSize.SMALL:
            shapes = (8, 16, 32, 64, 128)
        else:
            raise ValueError(f"Strange size for U2netEncoder: {size}")
        configuration: List[Union[RSUD1Config, RSUD5Config]] = [
            RSUD1Config(in_channels=in_channels, mid_channels=shapes[0], out_channels=shapes[1]),
            RSUD1Config(
                in_channels=shapes[1], mid_channels=shapes[0], out_channels=shapes[2], depth=4
            ),
            RSUD1Config(
                in_channels=shapes[2], mid_channels=shapes[1], out_channels=shapes[3], depth=3
            ),
            RSUD1Config(
                in_channels=shapes[3], mid_channels=shapes[2], out_channels=shapes[4], depth=2
            ),
            RSUD5Config(in_channels=shapes[4], mid_channels=shapes[3], out_channels=shapes[4]),
        ]
        downsample_configuration: List[DownsampleX2BlockConfig] = [
            DownsampleX2BlockConfig(in_channels=c.out_channels, max_pool=max_pool)
            for c in configuration
        ]
        self.encoders_stages = nn.ModuleList([c.create() for c in configuration])
        self.downsample_stages = nn.ModuleList([c.create() for c in downsample_configuration])

    def forward(self, batch: Tensor) -> Tuple[Tensor, List[Tensor]]:
        """Forward step of module"""
        encoder_outputs: List[Tensor] = []
        for encoder_stage, downsample_stage in zip(self.encoders_stages, self.downsample_stages):
            batch = downsample_stage(encoder_stage(batch))
            encoder_outputs.append(batch)
        return batch, encoder_outputs


class U2netBottleneck(ModuleWithDevice):
    """U2net bottleneck"""

    def __init__(self, size: ModelSize) -> None:
        super().__init__()
        shapes: Tuple[int, int]
        if size is ModelSize.BIG:
            shapes = (512, 256)
        elif size is ModelSize.MEDIUM:
            shapes = (256, 128)
        elif size is ModelSize.SMALL:
            shapes = (128, 64)
        else:
            raise ValueError(f"Strange size for U2netBottleneck: {size}")
        configuration = RSUD5Config(
            in_channels=shapes[0], mid_channels=shapes[1], out_channels=shapes[0]
        )
        self.bottleneck = configuration.create()

    def forward(self, batch: Tensor) -> Tensor:
        """Forward step of module"""
        return self.bottleneck(batch)


class U2netDecoder(ModuleWithDevice):
    """U2net decoder"""

    def __init__(self, size: ModelSize, bilinear: bool) -> None:
        super().__init__()
        if size is ModelSize.BIG:
            shapes = (1024, 512, 256, 128, 64, 32, 16)
        elif size is ModelSize.MEDIUM:
            shapes = (512, 256, 128, 64, 32, 16, 8)
        elif size is ModelSize.SMALL:
            shapes = (256, 128, 64, 32, 16, 8, 4)
        else:
            raise ValueError(f"Strange size for U2netDecoder: {size}")
        configuration: List[Union[RSUD5Config, RSUD1Config]] = [
            RSUD5Config(in_channels=shapes[0], mid_channels=shapes[2], out_channels=shapes[1]),
            RSUD1Config(
                in_channels=shapes[0], mid_channels=shapes[3], out_channels=shapes[2], depth=2
            ),
            RSUD1Config(
                in_channels=shapes[1], mid_channels=shapes[4], out_channels=shapes[3], depth=3
            ),
            RSUD1Config(
                in_channels=shapes[2], mid_channels=shapes[5], out_channels=shapes[4], depth=4
            ),
            RSUD1Config(in_channels=shapes[3], mid_channels=shapes[6], out_channels=shapes[4]),
        ]

        upsample_configuration: List[UpsampleX2BlockConfig] = [
            UpsampleX2BlockConfig(in_channels=c.in_channels, bilinear=bilinear)
            for c in configuration
        ]
        self.decoders_stages = nn.ModuleList([conf.create() for conf in configuration])
        self.upsample_stages = nn.ModuleList([conf.create() for conf in upsample_configuration])

    def forward(self, batch: Tensor, encoder_outputs) -> List[Tensor]:
        """Forward step of module"""
        decoder_outputs = []
        for decoder_stage, upsample_stage, encoder_output in zip(
            self.decoders_stages, self.upsample_stages, encoder_outputs[::-1]
        ):
            batch = decoder_stage(upsample_stage(torch.cat([batch, encoder_output], dim=1)))
            decoder_outputs.append(batch)
        return decoder_outputs


class U2netPredictHeads(ModuleWithDevice):
    """U2net predict heads"""

    def __init__(self, out_channels: int, size: ModelSize, count_predict_masks: int = 5) -> None:
        super().__init__()
        assert 0 < count_predict_masks < 6, "Count features from encoder maximum is 5"
        self.count_predict_masks = count_predict_masks
        in_channels: Tuple[int, int, int, int, int]
        if size is ModelSize.BIG:
            in_channels = (512, 256, 128, 64, 64)
        elif size is ModelSize.MEDIUM:
            in_channels = (256, 128, 64, 32, 32)
        elif size is ModelSize.SMALL:
            in_channels = (128, 64, 32, 16, 16)
        else:
            raise ValueError(f"Strange size for U2netPredictHead: {size}")
        configuration: List[Union[UpsamplePredictHeadConfig, DWConv2dConfig]] = [
            UpsamplePredictHeadConfig(
                scale_factor=16, in_channels=in_channels[-5], out_channels=out_channels
            ),
            UpsamplePredictHeadConfig(
                scale_factor=8, in_channels=in_channels[-4], out_channels=out_channels
            ),
            UpsamplePredictHeadConfig(
                scale_factor=4, in_channels=in_channels[-3], out_channels=out_channels
            ),
            UpsamplePredictHeadConfig(
                scale_factor=2, in_channels=in_channels[-2], out_channels=out_channels
            ),
            DWConv2dConfig(in_channels=in_channels[-1], out_channels=out_channels),
        ]
        self.heads = nn.ModuleList([c.create() for c in configuration])

    def forward(self, decoder_outputs: List[Tensor]) -> Tuple[Tensor, ...]:
        """Forward step of module"""
        headers_output: List[Tensor] = []
        for decoder_batch, clf_head in zip(
            decoder_outputs[-self.count_predict_masks :], self.heads[-self.count_predict_masks :]
        ):
            headers_output.append(clf_head(decoder_batch))
        return tuple(headers_output)


class U2net(ModuleWithDevice):
    """
    Implementation from https://arxiv.org/pdf/2005.09007.pdf
    U2net for multiple class or one class segmentation.
    Number of class can be changed by parametr 'out_channels'.
    """

    def __init__(  # pylint: disable=too-many-arguments
        self,
        in_channels: int,
        out_channels: int,
        size: ModelSize,
        max_pool: bool,
        bilinear: bool,
        count_predict_masks: int,
    ):
        """Module init"""
        super().__init__()
        self.encoder = U2netEncoder(in_channels=in_channels, size=size, max_pool=max_pool)
        self.bottleneck = U2netBottleneck(size=size)
        self.decoder = U2netDecoder(size=size, bilinear=bilinear)
        self.heads = U2netPredictHeads(
            out_channels=out_channels,
            size=size,
            count_predict_masks=count_predict_masks,
        )

    def forward(self, batch: Tensor) -> Tuple[Tensor, ...]:
        """Forward step of U2net"""
        batch, encoder_outputs = self.encoder.forward(batch=batch)
        batch = self.bottleneck.forward(batch=batch)
        decoder_outputs = self.decoder.forward(batch=batch, encoder_outputs=encoder_outputs)
        return self.heads.forward(decoder_outputs=decoder_outputs)
