from typing import List, Tuple, Union

import torch
from torch import Tensor, nn

from .blocks import ModuleWithDevice
from .configurations import (
    DownBlockConfig,
    DWConv2dBNLReLUConfig,
    DWConv2dSigmoidConfig,
    DWConv2dSigmoidUpConfig,
    ManyConfigs,
    UpBlockConfig,
)


class Unet(ModuleWithDevice):
    """Unet model https://arxiv.org/pdf/1505.04597.pdf"""

    def __init__(  # pylint: disable=too-many-arguments
        self,
        in_channels: int,
        out_channels: int,
        big: bool = True,
        max_pool: bool = True,
        bilinear: bool = True,
    ) -> None:
        """Model init"""
        super().__init__()
        if big:
            conf = [
                ManyConfigs(
                    [
                        DWConv2dBNLReLUConfig(in_channels=in_channels, out_channels=64),
                        DWConv2dBNLReLUConfig(in_channels=64, out_channels=64),
                    ]
                ),
                ManyConfigs(
                    [
                        DWConv2dBNLReLUConfig(in_channels=64, out_channels=128),
                        DWConv2dBNLReLUConfig(in_channels=128, out_channels=128),
                    ]
                ),
                ManyConfigs(
                    [
                        DWConv2dBNLReLUConfig(in_channels=128, out_channels=256),
                        DWConv2dBNLReLUConfig(in_channels=256, out_channels=256),
                    ]
                ),
                ManyConfigs(
                    [
                        DWConv2dBNLReLUConfig(in_channels=256, out_channels=512),
                        DWConv2dBNLReLUConfig(in_channels=512, out_channels=512),
                    ]
                ),
                ManyConfigs(
                    [
                        DWConv2dBNLReLUConfig(in_channels=512, out_channels=1024),
                        DWConv2dBNLReLUConfig(in_channels=1024, out_channels=512),
                    ]
                ),
                ManyConfigs(
                    [
                        DWConv2dBNLReLUConfig(in_channels=1024, out_channels=512),
                        DWConv2dBNLReLUConfig(in_channels=512, out_channels=256),
                    ]
                ),
                ManyConfigs(
                    [
                        DWConv2dBNLReLUConfig(in_channels=512, out_channels=256),
                        DWConv2dBNLReLUConfig(in_channels=256, out_channels=128),
                    ]
                ),
                ManyConfigs(
                    [
                        DWConv2dBNLReLUConfig(in_channels=256, out_channels=128),
                        DWConv2dBNLReLUConfig(in_channels=128, out_channels=64),
                    ]
                ),
                ManyConfigs(
                    [
                        DWConv2dBNLReLUConfig(in_channels=128, out_channels=64),
                        DWConv2dBNLReLUConfig(in_channels=64, out_channels=64),
                    ]
                ),
            ]
        else:
            conf = [
                ManyConfigs(
                    [
                        DWConv2dBNLReLUConfig(in_channels=in_channels, out_channels=32),
                        DWConv2dBNLReLUConfig(in_channels=32, out_channels=32),
                    ]
                ),
                ManyConfigs(
                    [
                        DWConv2dBNLReLUConfig(in_channels=32, out_channels=64),
                        DWConv2dBNLReLUConfig(in_channels=64, out_channels=64),
                    ]
                ),
                ManyConfigs(
                    [
                        DWConv2dBNLReLUConfig(in_channels=64, out_channels=128),
                        DWConv2dBNLReLUConfig(in_channels=128, out_channels=128),
                    ]
                ),
                ManyConfigs(
                    [
                        DWConv2dBNLReLUConfig(in_channels=128, out_channels=256),
                        DWConv2dBNLReLUConfig(in_channels=256, out_channels=256),
                    ]
                ),
                ManyConfigs(
                    [
                        DWConv2dBNLReLUConfig(in_channels=256, out_channels=512),
                        DWConv2dBNLReLUConfig(in_channels=512, out_channels=256),
                    ]
                ),
                ManyConfigs(
                    [
                        DWConv2dBNLReLUConfig(in_channels=512, out_channels=256),
                        DWConv2dBNLReLUConfig(in_channels=256, out_channels=128),
                    ]
                ),
                ManyConfigs(
                    [
                        DWConv2dBNLReLUConfig(in_channels=256, out_channels=128),
                        DWConv2dBNLReLUConfig(in_channels=128, out_channels=64),
                    ]
                ),
                ManyConfigs(
                    [
                        DWConv2dBNLReLUConfig(in_channels=128, out_channels=64),
                        DWConv2dBNLReLUConfig(in_channels=64, out_channels=32),
                    ]
                ),
                ManyConfigs(
                    [
                        DWConv2dBNLReLUConfig(in_channels=64, out_channels=32),
                        DWConv2dBNLReLUConfig(in_channels=32, out_channels=32),
                    ]
                ),
            ]
        down_conf: List[DownBlockConfig] = [
            DownBlockConfig(in_channels=c.configs[-1].out_channels, max_pool=max_pool)
            for c in conf[:4]
        ]
        up_conf: List[UpBlockConfig] = [
            UpBlockConfig(in_channels=c.configs[-1].out_channels, bilinear=bilinear)
            for c in conf[-5:-1]
        ]
        clf_heads_config: List[Union[DWConv2dSigmoidConfig, DWConv2dSigmoidUpConfig]] = [
            DWConv2dSigmoidUpConfig(
                in_channels=conf[-5].configs[-1].out_channels,
                out_channels=out_channels,
                scale=16,
            ),
            DWConv2dSigmoidUpConfig(
                in_channels=conf[-4].configs[-1].out_channels,
                out_channels=out_channels,
                scale=8,
            ),
            DWConv2dSigmoidUpConfig(
                in_channels=conf[-3].configs[-1].out_channels,
                out_channels=out_channels,
                scale=4,
            ),
            DWConv2dSigmoidUpConfig(
                in_channels=conf[-2].configs[-1].out_channels,
                out_channels=out_channels,
                scale=2,
            ),
            DWConv2dSigmoidConfig(
                in_channels=conf[-1].configs[-1].out_channels, out_channels=out_channels
            ),
        ]
        main_clf_head = DWConv2dSigmoidConfig(
            in_channels=out_channels * 5,
            out_channels=out_channels,
            kernel_size=1,
            padding=0,
        )

        self.encoder_stages = nn.ModuleList([c.create() for c in conf[:4]])
        self.down_stages = nn.ModuleList([c.create() for c in down_conf])
        self.dilation_stage = conf[4].create()
        self.up_stages = nn.ModuleList([c.create() for c in up_conf])
        self.decoder_stages = nn.ModuleList([c.create() for c in conf[-4:]])
        self.clf_heads = nn.ModuleList([conf.create() for conf in clf_heads_config])
        self.main_clf_head = main_clf_head.create()

    def forward(self, batch: Tensor) -> Tuple[Tensor, ...]:
        """Forward step of module"""
        encoder_outputs, decoder_outputs, headers_outputs = [], [], []
        for encoder_stage, down_stage in zip(self.encoder_stages, self.down_stages):
            batch = encoder_stage(batch)
            encoder_outputs.append(batch)
            batch = down_stage(batch)
        batch = self.dilation_stage(batch)
        decoder_outputs.append(batch)
        for encoder_batch, up_stage, decoder_stage in zip(
            encoder_outputs[::-1], self.up_stages, self.decoder_stages
        ):
            batch = decoder_stage(torch.cat([encoder_batch, up_stage(batch)], dim=1))
            decoder_outputs.append(batch)
        for decoder_batch, clf_head in zip(decoder_outputs, self.clf_heads):
            headers_outputs.append(clf_head(decoder_batch))
        headers_outputs.append(self.main_clf_head(torch.cat(headers_outputs, dim=1)))
        return tuple(headers_outputs)


__all__ = ["Unet"]
