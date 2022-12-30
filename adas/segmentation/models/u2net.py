from typing import List, Tuple

import torch
from torch import Tensor, nn

from adas.core.models.blocks import (
    RSUD1,
    RSUD5,
    DownsampleX2Block,
    DWConv2d,
    ModuleWithDevice,
    UpsampleBlock,
    UpsampleX2Block,
)

from .types import ModelSize


class U2netEncoder(ModuleWithDevice):
    """U2net encoder"""

    def __init__(self, in_channels: int, size: str, downsample_mode: str) -> None:
        super().__init__()
        self.size = size
        shapes: Tuple[int, int, int, int, int]
        if size == ModelSize.BIG.value:
            shapes = (32, 64, 128, 256, 512)
        elif size == ModelSize.MEDIUM.value:
            shapes = (16, 32, 64, 128, 256)
        elif size == ModelSize.SMALL.value:
            shapes = (8, 16, 32, 64, 128)
        else:
            acceptable = [_.value for _ in ModelSize]
            raise ValueError(f"Strange size for U2netEncoder: {size}. Acceptable: {acceptable}")
        self.encoders_stages = nn.ModuleList(
            [
                RSUD1(in_channels=in_channels, out_channels=shapes[1], depth=5),
                RSUD1(in_channels=shapes[1], out_channels=shapes[2], depth=4),
                RSUD1(in_channels=shapes[2], out_channels=shapes[3], depth=3),
                RSUD1(in_channels=shapes[3], out_channels=shapes[4], depth=2),
                RSUD5(in_channels=shapes[4], out_channels=shapes[4]),
            ]
        )
        self.downsample_stages = nn.ModuleList(
            [
                DownsampleX2Block(mode=downsample_mode, in_channels=in_channels)
                for in_channels in shapes[1:]
            ]
        )
        self.downsample_stages.append(
            DownsampleX2Block(mode=downsample_mode, in_channels=shapes[4])
        )

    def forward(self, batch: Tensor) -> Tuple[Tensor, List[Tensor]]:
        """Forward step of U2netEncoder module"""
        encoder_outputs: List[Tensor] = []
        for encoder_stage, downsample_stage in zip(self.encoders_stages, self.downsample_stages):
            batch = downsample_stage(encoder_stage(batch))
            encoder_outputs.append(batch)
        return batch, encoder_outputs


class U2netBottleneck(ModuleWithDevice):
    """U2net bottleneck"""

    def __init__(self, size: str) -> None:
        super().__init__()
        shape: int
        if size == ModelSize.BIG.value:
            shape = 512
        elif size == ModelSize.MEDIUM.value:
            shape = 256
        elif size == ModelSize.SMALL.value:
            shape = 128
        else:
            acceptable = [_.value for _ in ModelSize]
            raise ValueError(f"Strange size for U2netBottleneck: {size}. Acceptable: {acceptable}")
        self.bottleneck = RSUD5(in_channels=shape, out_channels=shape)

    def forward(self, batch: Tensor) -> Tensor:
        """Forward step of U2netBottleneck module"""
        return self.bottleneck(batch)


class U2netDecoder(ModuleWithDevice):
    """U2net decoder"""

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
            raise ValueError(f"Strange size for U2netDecoder: {size}. Acceptable: {acceptable}")

        self.decoders_stages = nn.ModuleList(
            [
                RSUD5(in_channels=shapes[0], out_channels=shapes[1]),
                RSUD1(in_channels=shapes[0], out_channels=shapes[2], depth=2),
                RSUD1(in_channels=shapes[1], out_channels=shapes[3], depth=3),
                RSUD1(in_channels=shapes[2], out_channels=shapes[4], depth=4),
                RSUD1(in_channels=shapes[3], out_channels=shapes[4], depth=5),
            ]
        )
        self.upsample_stages = nn.ModuleList(
            [
                UpsampleX2Block(mode=upsample_mode, in_channels=in_channels)
                for in_channels in [shapes[0]] + list(shapes[:-1])
            ]
        )

    def forward(self, batch: Tensor, encoder_outputs) -> List[Tensor]:
        """Forward step of U2netDecoder module"""
        decoder_outputs = []
        # assert (
        #     False
        # ), f"{len(self.decoders_stages)}, {len(self.upsample_stages)}, {len(encoder_outputs[::-1])}"
        for decoder_stage, upsample_stage, encoder_output in zip(
            self.decoders_stages, self.upsample_stages, encoder_outputs[::-1]
        ):
            batch = decoder_stage(upsample_stage(torch.cat([batch, encoder_output], dim=1)))
            decoder_outputs.append(batch)
        return decoder_outputs


class U2netPredictHeads(ModuleWithDevice):
    """U2net predict heads"""

    def __init__(self, out_channels: int, size: str, count_features: int = 5) -> None:
        super().__init__()
        assert 0 < count_features < 6, "Count features from encoder maximum is 5"
        self.count_features = count_features
        shapes: Tuple[int, int, int, int, int]
        if size == ModelSize.BIG.value:
            shapes = (512, 256, 128, 64, 64)
        elif size == ModelSize.MEDIUM.value:
            shapes = (256, 128, 64, 32, 32)
        elif size == ModelSize.SMALL.value:
            shapes = (128, 64, 32, 16, 16)
        else:
            acceptable = [_.value for _ in ModelSize]
            raise ValueError(
                f"Strange size for U2netPredictHeads: {size}. Acceptable: {acceptable}"
            )
        heads = [
            nn.Sequential(
                UpsampleBlock(scale_factor=16),
                DWConv2d(in_channels=shapes[0], out_channels=out_channels),
            ),
            nn.Sequential(
                UpsampleBlock(scale_factor=8),
                DWConv2d(in_channels=shapes[1], out_channels=out_channels),
            ),
            nn.Sequential(
                UpsampleBlock(scale_factor=4),
                DWConv2d(in_channels=shapes[2], out_channels=out_channels),
            ),
            nn.Sequential(
                UpsampleBlock(scale_factor=2),
                DWConv2d(in_channels=shapes[3], out_channels=out_channels),
            ),
            DWConv2d(in_channels=shapes[4], out_channels=out_channels),
        ]
        self.heads = nn.ModuleList(heads[-count_features:])

    def forward(self, decoder_outputs: List[Tensor]) -> Tuple[Tensor, ...]:
        """Forward step of U2netPredictHeads module"""
        headers_output: List[Tensor] = []
        for decoder_batch, clf_head in zip(decoder_outputs[-self.count_features :], self.heads):
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
        size: str,
        downsample_mode: str,
        upsample_mode: str,
        count_features: int,
    ):
        """Module init"""
        super().__init__()
        self.encoder = U2netEncoder(
            in_channels=in_channels, size=size, downsample_mode=downsample_mode
        )
        self.bottleneck = U2netBottleneck(size=size)
        self.decoder = U2netDecoder(size=size, upsample_mode=upsample_mode)
        self.heads = U2netPredictHeads(
            out_channels=out_channels, size=size, count_features=count_features
        )

    def forward(self, batch: Tensor) -> Tuple[Tensor, ...]:
        """Forward step of U2net model"""
        batch, encoder_outputs = self.encoder.forward(batch=batch)
        batch = self.bottleneck.forward(batch=batch)
        decoder_outputs = self.decoder.forward(batch=batch, encoder_outputs=encoder_outputs)
        return self.heads.forward(decoder_outputs=decoder_outputs)
