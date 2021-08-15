import torch

from torch import nn
from torch.nn import functional as F

from .u2net_blocks import blocks
from .utils import functions as utils


class U2Net(nn.Module):
    """
    Implementation from https://arxiv.org/pdf/2005.09007.pdf
    U2net for multiple class or one class segmentation.
    Number of class can be changed by parametr \'out_channels\'.
    """

    def __init__(self, in_channels: int, out_channels: int, is_big_net: bool = True):
        super().__init__()
        if is_big_net:
            configuration = [
                [32, 64],
                [64, 32, 128],
                [128, 64, 256],
                [256, 128, 512],
                [512, 256, 512],
                [512, 256, 512],
                [1024, 256, 512],
                [1024, 128, 256],
                [512, 64, 128],
                [256, 32, 64],
                [128, 16, 64]
            ]
        else:
            configuration = [
                [16, 64],
                [64, 16, 64],
                [64, 16, 64],
                [64, 16, 64],
                [64, 16, 64],
                [64, 16, 64],
                [128, 16, 64],
                [128, 16, 64],
                [128, 16, 64],
                [128, 16, 64],
                [128, 16, 64]
            ]

        # encoder
        self.encoder = nn.ModuleList(
            [
                blocks.RSUOneDilation([in_channels, *configuration[0]], depth=7),
                blocks.RSUOneDilation(configuration[1], depth=6),
                blocks.RSUOneDilation(configuration[2], depth=5),
                blocks.RSUOneDilation(configuration[3], depth=4),
                blocks.RSU4FiveDilation(configuration[4])
            ]
        )
        self.dilation = blocks.RSU4FiveDilation(configuration[5])
        # decoder
        self.decoder = nn.ModuleList(
            [
                blocks.RSU4FiveDilation(configuration[6]),
                blocks.RSUOneDilation(configuration[7], depth=4),
                blocks.RSUOneDilation(configuration[8], depth=5),
                blocks.RSUOneDilation(configuration[9], depth=6),
                blocks.RSUOneDilation(configuration[10], depth=7)
            ]
        )
        self.decoder_resulting = nn.ModuleList(
            [
                nn.Conv2d(configuration[-5][-1], out_channels, kernel_size=3, padding=1),
                nn.Conv2d(configuration[-4][-1], out_channels, kernel_size=3, padding=1),
                nn.Conv2d(configuration[-3][-1], out_channels, kernel_size=3, padding=1),
                nn.Conv2d(configuration[-2][-1], out_channels, kernel_size=3, padding=1),
                nn.Conv2d(configuration[-1][-1], out_channels, kernel_size=3, padding=1),
                nn.Conv2d(configuration[-6][-1], out_channels, kernel_size=3, padding=1),
                nn.Conv2d(out_channels * 6, out_channels, kernel_size=3, padding=1),

            ]
        )

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, img):
        output_ecoder = []
        output_decoder = []
        result_encoder = []
        upsample_output = []
        # encoder stage
        input_img = img.clone()
        for i in range(5):
            output_ecoder.append(
                self.encoder[i](input_img)
            )
            input_img = F.max_pool2d(
                output_ecoder[i],
                kernel_size=2,
                stride=2
            )
        # dilation part
        input_img = self.dilation(input_img)
        dilation_output = input_img.clone()
        # decoder stage
        for i in range(5):
            input_img = utils.upsample_like(input_img, output_ecoder[-i-1])
            output_decoder.append(
                self.decoder[i](
                    torch.cat(
                        (input_img, output_ecoder[-i-1]),
                        dim=1
                    )
                )
            )
            input_img = output_decoder[i]
        # output stage
        for i in range(5):
            upsample_output.append(
                utils.upsample_like(
                    output_decoder[i],
                    img
                )
            )
            result_encoder.append(
                self.decoder_resulting[i](upsample_output[i])
            )
        upsample_output.append(
            utils.upsample_like(dilation_output, img)
        )
        result_encoder.append(
            self.decoder_resulting[5](upsample_output[5])
        )
        result_encoder.append(
            self.decoder_resulting[6](
                torch.cat(
                    result_encoder,
                    dim=1
                )
            )
        )
        return [torch.sigmoid(x) for x in result_encoder]
