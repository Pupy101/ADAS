from typing import List, Tuple, Union

import torch
from torch import Tensor, nn

from .blocks import DownBlock, DWConv2dS, NDWConv2dBNLReLU, UpBlock, UpDWConv2dS
from .configurations import DownBlockConfig, DWConv2dSConfig, NDWConv2dBNLReLUConfig, UpBlockConfig, UpDWConv2dSConfig


class Unet(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        big: bool = True,
        max_pool: bool = True,
        bilinear: bool = True,
    ) -> None:
        super().__init__()
        conf: List[NDWConv2dBNLReLUConfig]
        if big:
            conf = [
                NDWConv2dBNLReLUConfig(NDWConv2dBNLReLU, in_channels=in_channels, out_channels=64, count=3),
                NDWConv2dBNLReLUConfig(NDWConv2dBNLReLU, in_channels=64, out_channels=128, count=3),
                NDWConv2dBNLReLUConfig(NDWConv2dBNLReLU, in_channels=128, out_channels=256, count=3),
                NDWConv2dBNLReLUConfig(NDWConv2dBNLReLU, in_channels=256, out_channels=512, count=3),
                NDWConv2dBNLReLUConfig(NDWConv2dBNLReLU, in_channels=1024, out_channels=512, count=3),
                NDWConv2dBNLReLUConfig(NDWConv2dBNLReLU, in_channels=512, out_channels=256, count=3),
                NDWConv2dBNLReLUConfig(NDWConv2dBNLReLU, in_channels=256, out_channels=128, count=3),
                NDWConv2dBNLReLUConfig(NDWConv2dBNLReLU, in_channels=128, out_channels=64, count=3),
            ]
        else:
            conf = [
                NDWConv2dBNLReLUConfig(NDWConv2dBNLReLU, in_channels=in_channels, out_channels=16, count=3),
                NDWConv2dBNLReLUConfig(NDWConv2dBNLReLU, in_channels=16, out_channels=32, count=3),
                NDWConv2dBNLReLUConfig(NDWConv2dBNLReLU, in_channels=32, out_channels=64, count=3),
                NDWConv2dBNLReLUConfig(NDWConv2dBNLReLU, in_channels=64, out_channels=128, count=3),
                NDWConv2dBNLReLUConfig(NDWConv2dBNLReLU, in_channels=256, out_channels=128, count=3),
                NDWConv2dBNLReLUConfig(NDWConv2dBNLReLU, in_channels=128, out_channels=64, count=3),
                NDWConv2dBNLReLUConfig(NDWConv2dBNLReLU, in_channels=64, out_channels=32, count=3),
                NDWConv2dBNLReLUConfig(NDWConv2dBNLReLU, in_channels=32, out_channels=16, count=3),
            ]
        down_conf: List[DownBlockConfig] = [
            DownBlockConfig(DownBlock, in_channels=c.out_channels, out_channels=c.out_channels, max_pool=max_pool)
            for c in conf[:4]
        ]
        up_conf: List[UpBlockConfig] = [
            UpBlockConfig(UpBlock, in_channels=c.in_channels, out_channels=c.out_channels, bilinear=bilinear)
            for c in conf[-4:]
        ]
        dilation_conf = NDWConv2dBNLReLUConfig(
            NDWConv2dBNLReLU, in_channels=conf[3].out_channels, out_channels=conf[4].in_channels, count=5
        )
        clf_heads_config: List[Union[UpDWConv2dSConfig, DWConv2dSConfig]] = [
            UpDWConv2dSConfig(UpDWConv2dS, in_channels=dilation_conf.out_channels, out_channels=out_channels, scale=16),
            UpDWConv2dSConfig(UpDWConv2dS, in_channels=conf[-4].out_channels, out_channels=out_channels, scale=8),
            UpDWConv2dSConfig(UpDWConv2dS, in_channels=conf[-3].out_channels, out_channels=out_channels, scale=4),
            UpDWConv2dSConfig(UpDWConv2dS, in_channels=conf[-2].out_channels, out_channels=out_channels, scale=2),
            DWConv2dSConfig(DWConv2dS, in_channels=conf[-1].out_channels, out_channels=out_channels),
        ]
        main_clf_head = DWConv2dSConfig(DWConv2dS, in_channels=out_channels * 5, out_channels=out_channels)

        self.encoder_stages = nn.ModuleList([c.create_module() for c in conf[:4]])
        self.down_stages = nn.ModuleList([c.create_module() for c in down_conf])
        self.dilation_stage = dilation_conf.create_module()
        self.up_stages = nn.ModuleList([c.create_module() for c in up_conf])
        self.decoder_stages = nn.ModuleList([c.create_module() for c in conf[-4:]])
        self.clf_heads = nn.ModuleList([conf.create_module() for conf in clf_heads_config])
        self.main_clf_head = main_clf_head.create_module()

    def forward(self, batch: Tensor) -> Tuple[Tensor, ...]:
        encoder_outputs, decoder_outputs, headers_outputs = [], [], []
        for encoder_stage, down_stage in zip(self.encoder_stages, self.down_stages):
            batch = encoder_stage(batch)
            encoder_outputs.append(batch)
            batch = down_stage(batch)
        batch = self.dilation_stage(batch)
        decoder_outputs.append(batch)
        for encoder_batch, up_stage, decoder_stage in zip(encoder_outputs[::-1], self.up_stages, self.decoder_stages):
            batch = decoder_stage(torch.cat([encoder_batch, up_stage(batch)], dim=1))
            decoder_outputs.append(batch)
        for decoder_batch, clf_head in zip(decoder_outputs, self.clf_heads):
            headers_outputs.append(clf_head(decoder_batch))
        headers_outputs.append(self.main_clf_head(torch.cat(headers_outputs, dim=1)))
        return tuple(headers_outputs)

    @property
    def device(self):
        return next(self.parameters()).device
