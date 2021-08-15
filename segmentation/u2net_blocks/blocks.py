import torch

from typing import List

from torch import nn
from torch.nn import functional as F

from ..utils.functions import upsample, upsample_like


class ConvBNReLU(nn.Module):
    """
    Unit with structure: Convolution2D + Batch Normalization + ReLU
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        padding: int = 1,
        dilation: int = 1
    ):
        super().__init__()
        self.convolution2d = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation
        )
        self.batch_normalization = nn.BatchNorm2d(out_channels)
        self.fn_activation = nn.ReLU()

    def forward(self, x):
        x = self.convolution2d(x)
        x = self.batch_normalization(x)
        x = self.fn_activation(x)
        return x


class UpsampleConv(nn.Module):
    """
    Unit with structure: Upsample to needed shape + Conv2d + Sigmoid
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        padding: int = 1,
    ):
        super().__init__()
        self.convolution = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=padding
        )

    def forward(self, x, shape):
        x = upsample(x, shape)
        x = self.convolution(x)
        return x


class RSUOneDilation(nn.Module):
    """
    Part of U2Net with structure similar vanilla UNet
    This NN is stage encoder1 and decoder5 in U2Net
    """

    def __init__(
        self,
        channels: List,
        kernel_size: int = 3,
        padding: int = 1,
        padding_dilation: int = 2,
        dilation: int = 2,
        depth: int = 7
    ):
        super().__init__()
        assert len(channels) == 3, 'Length list of channels must be equal 3'
        assert depth >= 3, 'Depth of RSU unit must be bigger or equal 3'
        self._depth = depth
        # encoder part
        self.down_stage = nn.ModuleList(
            [
                ConvBNReLU(channels[0], channels[2], kernel_size, padding),
                ConvBNReLU(channels[2], channels[1], kernel_size, padding),
                *[
                    ConvBNReLU(channels[1], channels[1], kernel_size, padding)
                    for _ in range(depth - 2)
                ]

            ]
        )
        # dilation part
        self.dilation_stage = ConvBNReLU(channels[1], channels[1], kernel_size, padding_dilation, dilation=dilation)
        # decoder part
        self.up_stage = nn.ModuleList(
            [
                *[
                    ConvBNReLU(channels[1] * 2, channels[1], kernel_size, padding)
                    for _ in range(depth - 2)
                ],
                ConvBNReLU(channels[1] * 2, channels[2], kernel_size, padding)
            ]
        )

    def forward(self, input_tensor):
        output_down_stage = []
        # downsampling part
        output_down_stage.append(
            self.down_stage[0](input_tensor)
        )
        x = output_down_stage[0]
        for i in range(1, self._depth - 1):
            output_down_stage.append(
                self.down_stage[i](x)
            )
            x = F.max_pool2d(
                output_down_stage[i],
                kernel_size=2,
                stride=2
            )
        output_down_stage.append(
            self.down_stage[-1](x)
        )
        # dilation part
        output_dilation_stage = self.dilation_stage(output_down_stage[-1])
        # upsample part

        output_up = self.up_stage[0](
            torch.cat(
                (output_dilation_stage, output_down_stage[-1]),
                dim=1
            )
        )
        output_up = upsample_like(output_up, output_down_stage[-2])
        for i in range(1, self._depth - 2):
            output_up = self.up_stage[i](
                torch.cat(
                    (output_up, output_down_stage[-i-1]),
                    dim=1
                )
            )
            output_up = upsample_like(output_up, output_down_stage[-i-2])
        output_up = self.up_stage[-1](
            torch.cat(
                (output_up, output_down_stage[1]),
                dim=1
            )
        )
        return output_up + output_down_stage[0]


class RSU4FiveDilation(nn.Module):
    """
    Part of U2Net with structure similar vanilla UNet
    This NN is stage encoder5 and decoder1 in U2Net
    """

    def __init__(
        self,
        channels: List,
        kernel_size: int = 3,
        padding: int = 1,
    ):
        super().__init__()

        assert len(channels) == 3, 'Length list of channels must be equal 3'

        self.down_stage = nn.ModuleList(
            [
                ConvBNReLU(channels[0], channels[2], kernel_size),
                ConvBNReLU(channels[2], channels[1], kernel_size, dilation=1, padding=1),
                ConvBNReLU(channels[1], channels[1], kernel_size, dilation=2, padding=2),
                ConvBNReLU(channels[1], channels[1], kernel_size, dilation=4, padding=4)                
            ]
        )
        self.dilation_stage = ConvBNReLU(channels[1], channels[1], kernel_size, dilation=8, padding=8)
        self.up_stage = nn.ModuleList(
            [
                ConvBNReLU(channels[1] * 2, channels[1], kernel_size, dilation=4, padding=4),
                ConvBNReLU(channels[1] * 2, channels[1], kernel_size, dilation=2, padding=2),
                ConvBNReLU(channels[1] * 2, channels[2], kernel_size)
            ]
        )

    def forward(self, input_tensor):
        output_down_stage = []
        # downsampling part
        output_down_stage.append(
            self.down_stage[0](input_tensor)
        )
        for i in range(3):
            output_down_stage.append(
                self.down_stage[i + 1](output_down_stage[i])
            )
        # dilation part
        output_dilation = self.dilation_stage(output_down_stage[-1])
        # upsample part
        output_up_stage = self.up_stage[0](
                torch.cat((output_dilation, output_down_stage[-1]), dim=1)
        )
        for i in range(1, 3):
            output_up_stage = self.up_stage[i](
                torch.cat(
                    (output_up_stage, output_down_stage[-i - 1]),
                    dim=1
                )
            )
        return output_up_stage + output_down_stage[0]
