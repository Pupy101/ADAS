from typing import List, Tuple, Union

import torch
from torch import Tensor, nn

from .blocks import RSUD1, RSUD5, DownBlock, DWConv2dS, UpBlock, UpDWConv2dS
from .configurations import DownBlockConfig, DWConv2dSConfig, RSUD1Config, RSUD5Config, UpBlockConfig, UpDWConv2dSConfig


class U2net(nn.Module):
    """
    Implementation from https://arxiv.org/pdf/2005.09007.pdf
    U2net for multiple class or one class segmentation.
    Number of class can be changed by parametr \'out_channels\'.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        big: bool = True,
        max_pool: bool = True,
        bilinear: bool = True,
    ):
        super().__init__()
        conf: List[Union[RSUD1Config, RSUD5Config]]
        if big:
            conf = [
                RSUD1Config(RSUD1, in_channels=in_channels, mid_channels=32, out_channels=64),
                RSUD1Config(RSUD1, in_channels=64, mid_channels=32, out_channels=128, depth=4),
                RSUD1Config(RSUD1, in_channels=128, mid_channels=64, out_channels=256, depth=3),
                RSUD1Config(RSUD1, in_channels=256, mid_channels=128, out_channels=512, depth=2),
                RSUD5Config(RSUD5, in_channels=512, mid_channels=256, out_channels=512),
                RSUD5Config(RSUD5, in_channels=512, mid_channels=256, out_channels=512),
                RSUD5Config(RSUD5, in_channels=1024, mid_channels=256, out_channels=512),
                RSUD1Config(RSUD1, in_channels=1024, mid_channels=128, out_channels=256, depth=2),
                RSUD1Config(RSUD1, in_channels=512, mid_channels=64, out_channels=128, depth=3),
                RSUD1Config(RSUD1, in_channels=256, mid_channels=32, out_channels=64, depth=4),
                RSUD1Config(RSUD1, in_channels=128, mid_channels=16, out_channels=64),
            ]
        else:
            conf = [
                RSUD1Config(RSUD1, in_channels=in_channels, mid_channels=4, out_channels=16),
                RSUD1Config(RSUD1, in_channels=16, mid_channels=8, out_channels=32, depth=4),
                RSUD1Config(RSUD1, in_channels=32, mid_channels=16, out_channels=64, depth=3),
                RSUD1Config(RSUD1, in_channels=64, mid_channels=32, out_channels=128, depth=2),
                RSUD5Config(RSUD5, in_channels=128, mid_channels=64, out_channels=128),
                RSUD5Config(RSUD5, in_channels=128, mid_channels=64, out_channels=128),
                RSUD5Config(RSUD5, in_channels=256, mid_channels=64, out_channels=128),
                RSUD1Config(RSUD1, in_channels=256, mid_channels=32, out_channels=64, depth=2),
                RSUD1Config(RSUD1, in_channels=128, mid_channels=16, out_channels=32, depth=3),
                RSUD1Config(RSUD1, in_channels=64, mid_channels=8, out_channels=16, depth=4),
                RSUD1Config(RSUD1, in_channels=32, mid_channels=4, out_channels=16),
            ]
        down_conf: List[DownBlockConfig] = [
            DownBlockConfig(
                DownBlock,
                in_channels=c.out_channels,
                mid_channels=c.mid_channels,
                out_channels=c.out_channels,
                max_pool=max_pool,
            )
            for c in conf[:5]
        ]
        up_conf: List[UpBlockConfig] = [
            UpBlockConfig(
                UpBlock,
                in_channels=c.in_channels,
                mid_channels=c.mid_channels,
                out_channels=c.in_channels,
                bilinear=bilinear,
            )
            for c in conf[-5:]
        ]
        clf_heads_config: List[Union[UpDWConv2dSConfig, DWConv2dSConfig]] = [
            UpDWConv2dSConfig(UpDWConv2dS, in_channels=conf[-6].out_channels, out_channels=out_channels, scale=32),
            UpDWConv2dSConfig(UpDWConv2dS, in_channels=conf[-5].out_channels, out_channels=out_channels, scale=16),
            UpDWConv2dSConfig(UpDWConv2dS, in_channels=conf[-4].out_channels, out_channels=out_channels, scale=8),
            UpDWConv2dSConfig(UpDWConv2dS, in_channels=conf[-3].out_channels, out_channels=out_channels, scale=4),
            UpDWConv2dSConfig(UpDWConv2dS, in_channels=conf[-2].out_channels, out_channels=out_channels, scale=2),
            DWConv2dSConfig(DWConv2dS, in_channels=conf[-1].out_channels, out_channels=out_channels),
        ]
        main_clf_head = DWConv2dSConfig(DWConv2dS, in_channels=out_channels * 6, out_channels=out_channels)

        # encoder
        self.encoders_stages = nn.ModuleList([conf.create_module() for conf in conf[:5]])
        self.downsample_stages = nn.ModuleList([conf.create_module() for conf in down_conf])
        # dilation stage
        self.dilation_stage = conf[5].create_module()
        # decoder
        self.decoders_stages = nn.ModuleList([conf.create_module() for conf in conf[-5:]])
        self.upsample_stages = nn.ModuleList([conf.create_module() for conf in up_conf])
        # clf heads
        self.clf_heads = nn.ModuleList([conf.create_module() for conf in clf_heads_config])
        # main clf head
        self.main_clf_head = main_clf_head.create_module()

    def forward(self, batch: Tensor) -> Tuple[Tensor, ...]:
        outputs_encoder, outputs_decoder, headers_output = [], [], []
        for encoder_stage, downsample_stage in zip(self.encoders_stages, self.downsample_stages):
            batch = downsample_stage(encoder_stage(batch))
            outputs_encoder.append(batch)
        batch = self.dilation_stage(batch)
        outputs_decoder.append(batch)
        for decoder_stage, upsample_stage, encoder_batch in zip(
            self.decoders_stages, self.upsample_stages, outputs_encoder[::-1]
        ):
            batch = decoder_stage(upsample_stage(torch.cat([batch, encoder_batch], dim=1)))
            outputs_decoder.append(batch)
        for decoder_batch, clf_head in zip(outputs_decoder, self.clf_heads):
            headers_output.append(clf_head(decoder_batch))
        main_output = self.main_clf_head(torch.cat(headers_output, dim=1))
        headers_output.append(main_output)
        return tuple(headers_output)

    @property
    def device(self):
        return next(self.parameters()).device
