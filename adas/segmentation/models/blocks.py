from typing import Optional, Tuple

import torch
from torch import Tensor, nn


class ModuleWithDevice(nn.Module):  # pylint: disable=abstract-method
    """Mixin class for method of device model"""

    @property
    def device(self):
        """Device of model"""
        return next(self.parameters()).device


class DWConv2d(nn.Module):
    """DepthWise convolution"""

    def __init__(  # pylint: disable=too-many-arguments
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        dilation: int = 1,
    ) -> None:
        """Block init"""
        super().__init__()
        self.dw_conv = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=in_channels,
                dilation=dilation,
            ),
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=1,
            ),
        )

    def forward(self, batch: Tensor) -> Tensor:
        """Forward step of module"""
        return self.dw_conv(batch)


class DWConvT2d(nn.Module):
    """DepthWise transpose convolution"""

    def __init__(  # pylint: disable=too-many-arguments
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 2,
        stride: int = 2,
        padding: int = 0,
        output_padding: int = 0,
        dilation: int = 1,
    ) -> None:
        """Block init"""
        super().__init__()
        self.dw_t_conv = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                output_padding=output_padding,
                groups=in_channels,
                dilation=dilation,
            ),
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=1,
            ),
        )

    def forward(self, batch: Tensor) -> Tensor:
        """Forward step of module"""
        return self.dw_t_conv(batch)


class DWConv2dBNLReLU(nn.Module):
    """DepthWise convolution with BatchNorm2d and LeakyReLU activation"""

    def __init__(  # pylint: disable=too-many-arguments
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        dilation: int = 1,
        negative_slope: float = 0.05,
    ) -> None:
        """Block init"""
        super().__init__()
        self.conv2d = DWConv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
        )
        self.batch_norm = nn.BatchNorm2d(num_features=out_channels)
        self.activation = nn.LeakyReLU(negative_slope=negative_slope)

    def forward(self, batch: Tensor) -> Tensor:
        """Forward step of module"""
        return self.activation(self.batch_norm(self.conv2d(batch)))


class DWConvT2dBNLReLU(nn.Module):
    """DepthWise transpose convolution with BatchNorm2d and LeakyReLU activation"""

    def __init__(  # pylint: disable=too-many-arguments
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 2,
        stride: int = 2,
        padding: int = 0,
        output_padding: int = 0,
        dilation: int = 1,
        negative_slope: float = 0.05,
    ) -> None:
        """Block init"""
        super().__init__()
        self.dw_t_conv = DWConvT2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            dilation=dilation,
        )
        self.batch_norm = nn.BatchNorm2d(num_features=out_channels)
        self.activation = nn.LeakyReLU(negative_slope=negative_slope)

    def forward(self, batch: Tensor) -> Tensor:
        """Forward step of module"""
        return self.activation(self.batch_norm(self.dw_t_conv(batch)))


class UpsampleX2Block(nn.Module):
    """
    Upsample block with idea from
    https://github.com/milesial/Pytorch-UNet/blob/67bf11b4db4c5f2891bd7e8e7f58bcde8ee2d2db/unet/unet_parts.py#L42
    """

    def __init__(self, in_channels: int, bilinear: bool = True) -> None:
        """Block init"""
        super().__init__()
        self.upsample_block: nn.Module  # for mypy
        if bilinear:
            self.upsample_block = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        else:
            self.upsample_block = DWConvT2dBNLReLU(
                in_channels=in_channels, out_channels=in_channels
            )

    def forward(self, batch: Tensor) -> Tensor:
        """Forward step of module"""
        return self.upsample_block(batch)


class UpsampleBlock(nn.Module):
    """
    Upsample block with idea from
    https://github.com/milesial/Pytorch-UNet/blob/67bf11b4db4c5f2891bd7e8e7f58bcde8ee2d2db/unet/unet_parts.py#L42
    """

    def __init__(self, scale_factor: int) -> None:
        """Block init"""
        super().__init__()
        self.upsample_block = nn.Upsample(
            scale_factor=scale_factor, mode="bilinear", align_corners=True
        )

    def forward(self, batch: Tensor) -> Tensor:
        """Forward step of module"""
        return self.upsample_block(batch)


class DownsampleX2Block(nn.Module):
    """Downsample block with max pooling or convolution with stride = 2"""

    def __init__(self, in_channels: int, max_pool: bool = True) -> None:
        """Block init"""
        super().__init__()
        self.downsample_block: nn.Module  # for mypy
        if max_pool:
            self.downsample_block = nn.MaxPool2d(kernel_size=2, stride=2)
        else:
            self.downsample_block = DWConv2dBNLReLU(
                in_channels=in_channels, out_channels=in_channels, stride=2
            )

    def forward(self, batch: Tensor) -> Tensor:
        """Forward step of module"""
        return self.downsample_block(batch)


class RSUD1(nn.Module):  # pylint: disable=too-many-instance-attributes
    """
    This block is stage en_(1-4) and de_(1-4) in U2Net from https://arxiv.org/pdf/2005.09007.pdf
    RSU block with one dilated convolution
    """

    def __init__(  # pylint: disable=too-many-arguments
        self,
        in_channels: int,
        out_channels: int,
        mid_channels: Optional[int] = None,
        kernel_size: int = 3,
        padding: int = 1,
        dilation: int = 2,
        pad_dilation: int = 2,
        depth: int = 5,
        max_pool: bool = True,
        bilinear: bool = True,
    ) -> None:
        """Block init"""
        super().__init__()
        if mid_channels is None:
            mid_channels = out_channels
        assert depth >= 1, "Depth of RSU unit must be bigger or equal 3"
        # 2 preprocessing conv
        self.preprocess_conv_stage_1 = DWConv2dBNLReLU(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=padding,
        )
        self.preprocess_conv_stage_2 = DWConv2dBNLReLU(
            in_channels=out_channels,
            out_channels=mid_channels,
            kernel_size=kernel_size,
            padding=padding,
        )
        # encoder stage
        self.encoder_stages = nn.ModuleList(
            [
                nn.Sequential(
                    DownsampleX2Block(
                        in_channels=mid_channels,
                        max_pool=max_pool,
                    ),
                    DWConv2dBNLReLU(
                        in_channels=mid_channels,
                        out_channels=mid_channels,
                        kernel_size=kernel_size,
                        padding=padding,
                    ),
                )
                for _ in range(depth)
            ]
        )
        # dilation part
        self.dilation_stage = DWConv2dBNLReLU(
            in_channels=mid_channels,
            out_channels=mid_channels,
            padding=pad_dilation,
            dilation=dilation,
        )
        self.post_dilation_conv = DWConv2dBNLReLU(
            in_channels=mid_channels * 2,
            out_channels=mid_channels,
            kernel_size=kernel_size,
            padding=padding,
        )
        # decoder part
        self.decoder_upsample_stages = nn.ModuleList(
            [
                UpsampleX2Block(
                    in_channels=mid_channels,
                    bilinear=bilinear,
                )
                for _ in range(depth - 1)
            ]
        )
        self.decoder_conv_stages = nn.ModuleList(
            [
                DWConv2dBNLReLU(
                    in_channels=mid_channels * 2,
                    out_channels=mid_channels,
                    kernel_size=kernel_size,
                    padding=padding,
                )
                for _ in range(depth - 1)
            ]
        )
        # last decoder upsample & conv stage
        self.post_upsample_stage = UpsampleX2Block(
            in_channels=mid_channels,
            bilinear=bilinear,
        )
        self.post_conv_stage = DWConv2dBNLReLU(
            in_channels=mid_channels * 2,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=padding,
        )

    def forward(self, batch: Tensor) -> Tensor:
        """Forward step of module"""
        preprocess_batch_stage_1 = self.preprocess_conv_stage_1(batch)
        preprocess_batch_stage_2 = self.preprocess_conv_stage_2(preprocess_batch_stage_1)
        downsample_batches, current_batch = [], preprocess_batch_stage_2
        for encoder_stage in self.encoder_stages:
            current_batch = encoder_stage(current_batch)
            downsample_batches.append(current_batch)
        current_batch = self.post_dilation_conv(
            torch.cat([self.dilation_stage(current_batch), downsample_batches.pop()], dim=1)
        )
        for downsample_batch, upsample_stage, conv_stage in zip(
            downsample_batches[::-1],
            self.decoder_upsample_stages,
            self.decoder_conv_stages,
        ):
            current_batch = conv_stage(
                torch.cat([upsample_stage(current_batch), downsample_batch], dim=1)
            )

        current_batch = self.post_conv_stage(
            torch.cat([self.post_upsample_stage(current_batch), preprocess_batch_stage_2], dim=1)
        )
        return current_batch + preprocess_batch_stage_1


class RSUD5(nn.Module):  # pylint: disable=too-many-instance-attributes
    """
    Part of U2Net with structure similar vanilla UNet.
    This NN is stage en_5, en_6 and de_5 in U2Net
    """

    def __init__(  # pylint: disable=too-many-arguments
        self,
        in_channels: int,
        out_channels: int,
        mid_channels: Optional[int] = None,
        kernel_size: int = 3,
        padding: int = 1,
        dilations: Tuple[int, int, int, int, int] = (2, 4, 8, 4, 2),
        pad_dilations: Tuple[int, int, int, int, int] = (2, 4, 8, 4, 2),
    ) -> None:
        """Block init"""
        super().__init__()
        assert len(dilations) == 5
        assert len(pad_dilations) == 5
        if mid_channels is None:
            mid_channels = out_channels
        self.preprocess_conv_stage_1 = DWConv2dBNLReLU(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=padding,
        )
        self.preprocess_conv_stage_2 = DWConv2dBNLReLU(
            in_channels=out_channels,
            out_channels=mid_channels,
            kernel_size=kernel_size,
            padding=padding,
        )
        dil_kwargs = {"out_channels": mid_channels, "kernel_size": kernel_size}
        self.dilation_stage_1 = DWConv2dBNLReLU(
            in_channels=mid_channels,
            padding=pad_dilations[0],
            dilation=dilations[0],
            **dil_kwargs,
        )
        self.dilation_stage_2 = DWConv2dBNLReLU(
            in_channels=mid_channels,
            padding=pad_dilations[1],
            dilation=dilations[1],
            **dil_kwargs,
        )
        self.dilation_stage_3 = DWConv2dBNLReLU(
            in_channels=mid_channels,
            padding=pad_dilations[2],
            dilation=dilations[2],
            **dil_kwargs,
        )
        self.dilation_stage_4 = DWConv2dBNLReLU(
            in_channels=mid_channels * 2,
            padding=pad_dilations[3],
            dilation=dilations[3],
            **dil_kwargs,
        )
        self.dilation_stage_5 = DWConv2dBNLReLU(
            in_channels=mid_channels * 2,
            padding=pad_dilations[4],
            dilation=dilations[4],
            **dil_kwargs,
        )
        self.postprocessing_stage = DWConv2dBNLReLU(
            in_channels=mid_channels * 2,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=padding,
        )

    def forward(self, batch: Tensor) -> Tensor:
        """Forward step of module"""
        preprocessed_batch_1 = self.preprocess_conv_stage_1(batch)
        preprocessed_batch_2 = self.preprocess_conv_stage_2(preprocessed_batch_1)
        dil_out_1 = self.dilation_stage_1(preprocessed_batch_2)
        dil_out_2 = self.dilation_stage_2(dil_out_1)
        dil_out_3 = self.dilation_stage_3(dil_out_2)
        dil_out_4 = self.dilation_stage_4(torch.cat([dil_out_3, dil_out_2], dim=1))
        dil_out_5 = self.dilation_stage_5(torch.cat([dil_out_4, dil_out_1], dim=1))
        post_process_batch = self.postprocessing_stage(
            torch.cat([preprocessed_batch_2, dil_out_5], dim=1)
        )
        return post_process_batch + preprocessed_batch_1


__all__ = [
    "DWConv2d",
    "DWConv2dBNLReLU",
    "DWConvT2d",
    "DWConvT2dBNLReLU",
    "UpsampleX2Block",
    "UpsampleBlock",
    "DownsampleX2Block",
    "RSUD1",
    "RSUD5",
]
