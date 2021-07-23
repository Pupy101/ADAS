import torch

from torch import nn
from torch.nn import functional as F


class ConvBNReLU(nn.Module):
    """
    Unit with structure: Convolution2D + Batch Normalization + ReLU
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, padding: int = 1, dilation: int = 1):

        super().__init__()
        self.convolution2d = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, dilation=dilation)
        self.batch_normalization = nn.BatchNorm2d(out_channels)
        self.fn_activation = nn.ReLU()

    def forward(self, x):
        x = self.convolution2d(x)
        x = self.batch_normalization(x)
        x = self.fn_activation(x)
        return x


class UpsampleConvSigmoid(nn.Module):
    """
    Unit with structure: Upsample + Conv2d + Sigmoid
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):

        super().__init__()

        self.convolution = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)
        self.fn_activation = nn.Sigmoid()

    def forward(self, x, shape):
        x = F.interpolate(x, size=shape, mode='bilinear', align_corners=False)
        x = self.convolution(x)
        x = self.fn_activation(x)
        return x


class RSU7OneDilation(nn.Module):
    """
    Part of U2Net with structure similar vanilla UNet
    This NN is stage encoder1 and decoder5 in U2Net
    """

    def __init__(self, channels: list, kernel_size: int = 3, padding: int = 1, padding_dilation: int = 2,
                 dilation: int = 2):
        super().__init__()

        assert len(channels) == 4, 'Length list of channels must be equal 4'

        self.down_stage1 = ConvBNReLU(channels[0], channels[1], kernel_size, padding)
        self.down_stage2 = ConvBNReLU(channels[1], channels[2], kernel_size, padding)
        self.down_stage3 = ConvBNReLU(channels[2], channels[2], kernel_size, padding)
        self.down_stage4 = ConvBNReLU(channels[2], channels[2], kernel_size, padding)
        self.down_stage5 = ConvBNReLU(channels[2], channels[2], kernel_size, padding)
        self.down_stage6 = ConvBNReLU(channels[2], channels[2], kernel_size, padding)
        self.down_stage7 = ConvBNReLU(channels[2], channels[2], kernel_size, padding)

        self.dilation_stage = ConvBNReLU(channels[2], channels[2], kernel_size, padding_dilation, dilation=dilation)

        self.up_stage1 = ConvBNReLU(channels[2] * 2, channels[2], kernel_size, padding)
        self.up_stage2 = ConvBNReLU(channels[2] * 2, channels[2], kernel_size, padding)
        self.up_stage3 = ConvBNReLU(channels[2] * 2, channels[2], kernel_size, padding)
        self.up_stage4 = ConvBNReLU(channels[2] * 2, channels[2], kernel_size, padding)
        self.up_stage5 = ConvBNReLU(channels[2] * 2, channels[2], kernel_size, padding)
        self.up_stage6 = ConvBNReLU(channels[2] * 2, channels[3], kernel_size, padding)

    def forward(self, input_tensor):

        # downsampling part
        stage1_down = self.down_stage1(input_tensor)
        stage2_down = self.down_stage2(stage1_down)

        stage3_down, indices1 = F.max_pool2d(stage2_down, kernel_size=2, stride=2, return_indices=True)
        stage3_down = self.down_stage3(stage3_down)

        stage4_down, indices2 = F.max_pool2d(stage3_down, kernel_size=2, stride=2, return_indices=True)
        stage4_down = self.down_stage4(stage4_down)

        stage5_down, indices3 = F.max_pool2d(stage4_down, kernel_size=2, stride=2, return_indices=True)
        stage5_down = self.down_stage5(stage5_down)

        stage6_down, indices4 = F.max_pool2d(stage5_down, kernel_size=2, stride=2, return_indices=True)
        stage6_down = self.down_stage6(stage6_down)

        stage7_down, indices5 = F.max_pool2d(stage6_down, kernel_size=2, stride=2, return_indices=True)
        stage7_down = self.down_stage7(stage7_down)

        # dilation part
        output_dilation = self.dilation_stage(stage7_down)

        # upsample part
        stage1_up = self.up_stage1(torch.cat((output_dilation, stage7_down), dim=1))

        stage2_up = F.max_unpool2d(stage1_up, indices5, kernel_size=2, stride=2)
        stage2_up = self.up_stage2(torch.cat((stage2_up, stage6_down), dim=1))

        stage3_up = F.max_unpool2d(stage2_up, indices4, kernel_size=2, stride=2)
        stage3_up = self.up_stage3(torch.cat((stage3_up, stage5_down), dim=1))

        stage4_up = F.max_unpool2d(stage3_up, indices3, kernel_size=2, stride=2)
        stage4_up = self.up_stage4(torch.cat((stage4_up, stage4_down), dim=1))

        stage5_up = F.max_unpool2d(stage4_up, indices2, kernel_size=2, stride=2)
        stage5_up = self.up_stage5(torch.cat((stage5_up, stage3_down), dim=1))

        stage6_up = F.max_unpool2d(stage5_up, indices1, kernel_size=2, stride=2)
        stage6_up = self.up_stage6(torch.cat((stage6_up, stage2_down), dim=1))

        return stage6_up + stage1_down


class RSU6OneDilation(nn.Module):
    """
    Part of U2Net with structure similar vanilla UNet
    This NN is stage encoder2 and decoder4 in U2Net
    """

    def __init__(self,
                 channels: list,
                 kernel_size: int,
                 padding: int,
                 padding_dilation: int,
                 dilation: int):
        super().__init__()

        assert len(channels) == 4, 'Length list of channels must be equal 4'

        self.down_stage1 = ConvBNReLU(channels[0], channels[1], kernel_size, padding)
        self.down_stage2 = ConvBNReLU(channels[1], channels[2], kernel_size, padding)
        self.down_stage3 = ConvBNReLU(channels[2], channels[2], kernel_size, padding)
        self.down_stage4 = ConvBNReLU(channels[2], channels[2], kernel_size, padding)
        self.down_stage5 = ConvBNReLU(channels[2], channels[2], kernel_size, padding)
        self.down_stage6 = ConvBNReLU(channels[2], channels[2], kernel_size, padding)

        self.dilation_stage = ConvBNReLU(channels[2], channels[2], kernel_size, padding_dilation, dilation=dilation)

        self.up_stage1 = ConvBNReLU(channels[2] * 2, channels[2], kernel_size, padding)
        self.up_stage2 = ConvBNReLU(channels[2] * 2, channels[2], kernel_size, padding)
        self.up_stage3 = ConvBNReLU(channels[2] * 2, channels[2], kernel_size, padding)
        self.up_stage4 = ConvBNReLU(channels[2] * 2, channels[2], kernel_size, padding)
        self.up_stage5 = ConvBNReLU(channels[2] * 2, channels[3], kernel_size, padding)

    def forward(self, input_tensor):

        # downsampling part
        stage1_down = self.down_stage1(input_tensor)
        stage2_down = self.down_stage2(stage1_down)

        stage3_down, indices1 = F.max_pool2d(stage2_down, kernel_size=2, stride=2, return_indices=True)
        stage3_down = self.down_stage3(stage3_down)

        stage4_down, indices2 = F.max_pool2d(stage3_down, kernel_size=2, stride=2, return_indices=True)
        stage4_down = self.down_stage4(stage4_down)

        stage5_down, indices3 = F.max_pool2d(stage4_down, kernel_size=2, stride=2, return_indices=True)
        stage5_down = self.down_stage5(stage5_down)

        stage6_down, indices4 = F.max_pool2d(stage5_down, kernel_size=2, stride=2, return_indices=True)
        stage6_down = self.down_stage6(stage6_down)

        # dilation part
        output_dilation = self.dilation_stage(stage6_down)

        # upsample part
        stage1_up = self.up_stage1(torch.cat((output_dilation, stage6_down), dim=1))

        stage2_up = F.max_unpool2d(stage1_up, indices4, kernel_size=2, stride=2)
        stage2_up = self.up_stage2(torch.cat((stage2_up, stage5_down), dim=1))

        stage3_up = F.max_unpool2d(stage2_up, indices3, kernel_size=2, stride=2)
        stage3_up = self.up_stage3(torch.cat((stage3_up, stage4_down), dim=1))

        stage4_up = F.max_unpool2d(stage3_up, indices2, kernel_size=2, stride=2)
        stage4_up = self.up_stage4(torch.cat((stage4_up, stage3_down), dim=1))

        stage5_up = F.max_unpool2d(stage4_up, indices1, kernel_size=2, stride=2)
        stage5_up = self.up_stage5(torch.cat((stage5_up, stage2_down), dim=1))

        return stage5_up + stage1_down


class RSU5OneDilation(nn.Module):
    """
    Part of U2Net with structure similar vanilla UNet
    This NN is stage encoder3 and decoder3 in U2Net
    """

    def __init__(self,
                 channels: list,
                 kernel_size: int,
                 padding: int,
                 padding_dilation: int,
                 dilation: int):
        super().__init__()

        assert len(channels) == 4, 'Length list of channels must be equal 4'

        self.down_stage1 = ConvBNReLU(channels[0], channels[1], kernel_size, padding)
        self.down_stage2 = ConvBNReLU(channels[1], channels[2], kernel_size, padding)
        self.down_stage3 = ConvBNReLU(channels[2], channels[2], kernel_size, padding)
        self.down_stage4 = ConvBNReLU(channels[2], channels[2], kernel_size, padding)
        self.down_stage5 = ConvBNReLU(channels[2], channels[2], kernel_size, padding)

        self.dilation_stage = ConvBNReLU(channels[2], channels[2], kernel_size, padding_dilation, dilation=dilation)

        self.up_stage1 = ConvBNReLU(channels[2] * 2, channels[2], kernel_size, padding)
        self.up_stage2 = ConvBNReLU(channels[2] * 2, channels[2], kernel_size, padding)
        self.up_stage3 = ConvBNReLU(channels[2] * 2, channels[2], kernel_size, padding)
        self.up_stage4 = ConvBNReLU(channels[2] * 2, channels[3], kernel_size, padding)

    def forward(self, input_tensor):

        # downsampling part
        stage1_down = self.down_stage1(input_tensor)
        stage2_down = self.down_stage2(stage1_down)

        stage3_down, indices1 = F.max_pool2d(stage2_down, kernel_size=2, stride=2, return_indices=True)
        stage3_down = self.down_stage3(stage3_down)

        stage4_down, indices2 = F.max_pool2d(stage3_down, kernel_size=2, stride=2, return_indices=True)
        stage4_down = self.down_stage4(stage4_down)

        stage5_down, indices3 = F.max_pool2d(stage4_down, kernel_size=2, stride=2, return_indices=True)
        stage5_down = self.down_stage5(stage5_down)

        # dilation part
        output_dilation = self.dilation_stage(stage5_down)

        # upsample part
        stage1_up = self.up_stage1(torch.cat((output_dilation, stage5_down), dim=1))

        stage2_up = F.max_unpool2d(stage1_up, indices3, kernel_size=2, stride=2)
        stage2_up = self.up_stage2(torch.cat((stage2_up, stage4_down), dim=1))

        stage3_up = F.max_unpool2d(stage2_up, indices2, kernel_size=2, stride=2)
        stage3_up = self.up_stage3(torch.cat((stage3_up, stage3_down), dim=1))

        stage4_up = F.max_unpool2d(stage3_up, indices1, kernel_size=2, stride=2)
        stage4_up = self.up_stage4(torch.cat((stage4_up, stage2_down), dim=1))

        return stage4_up + stage1_down


class RSU4OneDilation(nn.Module):
    """
    Part of U2Net with structure similar vanilla UNet
    This NN is stage encoder4 and decoder2 in U2Net
    """

    def __init__(self,
                 channels: list,
                 kernel_size: int,
                 padding: int,
                 padding_dilation: int,
                 dilation: int):
        super().__init__()

        assert len(channels) == 4, 'Length list of channels must be equal 4'

        self.down_stage1 = ConvBNReLU(channels[0], channels[1], kernel_size, padding)
        self.down_stage2 = ConvBNReLU(channels[1], channels[2], kernel_size, padding)
        self.down_stage3 = ConvBNReLU(channels[2], channels[2], kernel_size, padding)
        self.down_stage4 = ConvBNReLU(channels[2], channels[2], kernel_size, padding)

        self.dilation_stage = ConvBNReLU(channels[2], channels[2], kernel_size, padding_dilation, dilation=dilation)

        self.up_stage1 = ConvBNReLU(channels[2] * 2, channels[2], kernel_size, padding)
        self.up_stage2 = ConvBNReLU(channels[2] * 2, channels[2], kernel_size, padding)
        self.up_stage3 = ConvBNReLU(channels[2] * 2, channels[3], kernel_size, padding)

    def forward(self, input_tensor):

        # downsampling part
        stage1_down = self.down_stage1(input_tensor)
        stage2_down = self.down_stage2(stage1_down)

        stage3_down, indices1 = F.max_pool2d(stage2_down, kernel_size=2, stride=2, return_indices=True)
        stage3_down = self.down_stage3(stage3_down)

        stage4_down, indices2 = F.max_pool2d(stage3_down, kernel_size=2, stride=2, return_indices=True)
        stage4_down = self.down_stage4(stage4_down)

        # dilation part
        output_dilation = self.dilation_stage(stage4_down)

        # upsample part
        stage1_up = self.up_stage1(torch.cat((output_dilation, stage4_down), dim=1))

        stage2_up = F.max_unpool2d(stage1_up, indices2, kernel_size=2, stride=2)
        stage2_up = self.up_stage2(torch.cat((stage2_up, stage3_down), dim=1))

        stage3_up = F.max_unpool2d(stage2_up, indices1, kernel_size=2, stride=2)
        stage3_up = self.up_stage3(torch.cat((stage3_up, stage2_down), dim=1))

        return stage3_up + stage1_down


class RSU4FiveDilation(nn.Module):
    """
    Part of U2Net with structure similar vanilla UNet
    This NN is stage encoder5 and decoder1 in U2Net
    """

    def __init__(self,
                 channels: list,
                 kernel_size: int,
                 padding: int,
                 padding_dilation: list,
                 dilation: list):
        super().__init__()

        assert len(channels) == 4, 'Length list of channels must be equal 4'
        assert len(padding_dilation) == len(dilation) and len(dilation) == 5, 'Lenght'

        self.down_stage1 = ConvBNReLU(channels[0], channels[1], kernel_size, padding)
        self.down_stage2 = ConvBNReLU(channels[1], channels[2], kernel_size, padding)
        self.down_stage3 = ConvBNReLU(channels[2], channels[2], kernel_size, padding=padding_dilation[0],
                                      dilation=dilation[0])
        self.down_stage4 = ConvBNReLU(channels[2], channels[2], kernel_size, padding=padding_dilation[1],
                                      dilation=dilation[1])

        self.dilation_stage = ConvBNReLU(channels[2], channels[2], kernel_size, padding=padding_dilation[2],
                                         dilation=dilation[2])

        self.up_stage1 = ConvBNReLU(channels[2] * 2, channels[2], kernel_size, padding=padding_dilation[3],
                                    dilation=dilation[3])
        self.up_stage2 = ConvBNReLU(channels[2] * 2, channels[2], kernel_size, padding=padding_dilation[4],
                                    dilation=dilation[4])
        self.up_stage3 = ConvBNReLU(channels[2] * 2, channels[3], kernel_size, padding)

    def forward(self, input_tensor):

        # downsampling part
        stage1_down = self.down_stage1(input_tensor)
        stage2_down = self.down_stage2(stage1_down)
        stage3_down = self.down_stage3(stage2_down)
        stage4_down = self.down_stage4(stage3_down)

        # dilation part
        output_dilation = self.dilation_stage(stage4_down)

        # upsample part
        stage1_up = self.up_stage1(torch.cat((output_dilation, stage4_down), dim=1))
        stage2_up = self.up_stage2(torch.cat((stage1_up, stage3_down), dim=1))
        stage3_up = self.up_stage3(torch.cat((stage2_up, stage2_down), dim=1))

        return stage3_up + stage1_down
