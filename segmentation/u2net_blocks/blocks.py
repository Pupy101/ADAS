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


class UpsampleConvSigmoid(nn.Module):
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
        self.fn_activation = nn.Sigmoid()

    def forward(self, x, shape):
        x = upsample(x, shape)
        x = self.convolution(x)
        x = self.fn_activation(x)
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
        assert len(channels) == 4, 'Length list of channels must be equal 4'
        assert depth >= 3, 'Depth of RSU unit must be bigger or equal 3'
        self._depth = depth
        self.down_stage = nn.ModuleList(
            [
                ConvBNReLU(channels[0], channels[1], kernel_size, padding),
                ConvBNReLU(channels[1], channels[2], kernel_size, padding),
                *[
                    ConvBNReLU(channels[2], channels[2], kernel_size, padding)
                    for _ in range(depth - 2)
                ]

            ]
        )
        self.dilation_stage = ConvBNReLU(channels[2], channels[2], kernel_size, padding_dilation, dilation=dilation)
        self.up_stage = nn.ModuleList(
            [
                *[
                    ConvBNReLU(channels[2] * 2, channels[2], kernel_size, padding)
                    for _ in range(depth - 2)
                ],
                ConvBNReLU(channels[2] * 2, channels[3], kernel_size, padding)
            ]
        )

    def forward(self, input_tensor):
        output_down_stage = []
        output_up_stage = []        
        # downsampling part
        # stage1_down  -  index 0
        output_down_stage.append(self.down_stage[0](input_tensor))
        # stage2_down  -  index 1
        output_down_stage.append(self.down_stage[1](output_down_stage[0]))
        for i in range(self._depth - 2):
            output_down_stage.append(
                self.down_stage[i+2](
                    F.max_pool2d(
                        output_down_stage[i+1],
                        kernel_size=2,
                        stride=2
                    )
                )
            )
        # dilation part
        output_dilation_stage = self.dilation_stage(output_down_stage[-1])
        # upsample part
        output_up_stage.append(
            self.up_stage[0](
                torch.cat(
                    (output_dilation_stage, output_down_stage[-1]),
                    dim=1
                )
            )
        )
        for i in range(self._depth - 2):
            output_up_stage.append(
                self.up_stage[i+1](
                    torch.cat(
                        (
                            upsample_like(
                                output_up_stage[0],
                                output_down_stage[self._depth-2-i]
                            ),
                            output_down_stage[self._depth-2-i]
                        ),
                        dim=1
                    )
                )
            )
        return output_up_stage[-1] + output_down_stage[0]


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
        padding_dilation: List = [2, 4, 8, 4, 2],
        dilation: List = [2, 4, 8, 4, 2]
    ):
        super().__init__()

        assert len(channels) == 4, 'Length list of channels must be equal 4'
        assert len(padding_dilation) == len(dilation) and len(dilation) == 5, 'Lenght padding_dilation must be equal dilation'

        self.down_stage = nn.ModuleList(
            [
                ConvBNReLU(channels[0], channels[1], kernel_size, padding),
                ConvBNReLU(channels[1], channels[2], kernel_size, padding),
                ConvBNReLU(channels[2], channels[2], kernel_size,
                           padding=padding_dilation[0], dilation=dilation[0]),
                ConvBNReLU(channels[2], channels[2], kernel_size,
                           padding=padding_dilation[1], dilation=dilation[1])
            ]
        )
        self.dilation_stage = ConvBNReLU(channels[2], channels[2], kernel_size,
                                         padding=padding_dilation[2], dilation=dilation[2])
        self.up_stage = nn.ModuleList(
            [
                ConvBNReLU(channels[2] * 2, channels[2], kernel_size,
                           padding=padding_dilation[3], dilation=dilation[3]),
                ConvBNReLU(channels[2] * 2, channels[2], kernel_size,
                           padding=padding_dilation[4], dilation=dilation[4]),
                ConvBNReLU(channels[2] * 2, channels[3], kernel_size, padding)
            ]
        )

    def forward(self, input_tensor):
        output_down_stage = []
        output_up_stage = []
        # downsampling part
        output_down_stage.append(self.down_stage[0](input_tensor))
        for i in range(3):
            output_down_stage.append(
                self.down_stage[i + 1](output_down_stage[i])
            )
        # dilation part
        output_dilation = self.dilation_stage(output_down_stage[-1])
        # upsample part
        output_up_stage.append(
            self.up_stage[0](
                torch.cat((output_dilation, output_down_stage[3]), dim=1)
            )
        )
        for i in range(2):
            output_up_stage.append(
                self.up_stage[i+1](
                    torch.cat((
                        output_up_stage[i], output_down_stage[2-i]
                        ),
                        dim=1
                    )
                )
            )
        return output_up_stage[-1] + output_down_stage[0]
