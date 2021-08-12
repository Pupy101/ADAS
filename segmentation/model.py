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
    This NN equal U2Net from paper, but I drop one block in center of net - En_6.
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        # encoder
        self.encoder = nn.ModuleList(
            [
                blocks.RSUOneDilation([in_channels, 8, 32, 8], depth=7),
                blocks.RSUOneDilation([8, 16, 64, 16], depth=6),
                blocks.RSUOneDilation([16, 32, 128, 32], depth=5),
                blocks.RSUOneDilation([32, 64, 256, 64], depth=4),
                blocks.RSU4FiveDilation([64, 128, 512, 128])
            ]
        )
        # decoder
        self.decoder = nn.ModuleList(
            [
                blocks.RSU4FiveDilation([128, 64, 512, 64]),
                blocks.RSUOneDilation([64 * 2, 32, 256, 32], depth=4),
                blocks.RSUOneDilation([32 * 2, 16, 128, 16], depth=5),
                blocks.RSUOneDilation([16 * 2, 8, 64, 8], depth=6),
                blocks.RSUOneDilation([8 * 2, 8, 32, 8], depth=7)
            ]
        )
        self.decoder_resulting = nn.ModuleList(
            [
                blocks.UpsampleConvSigmoid(64, out_channels),
                blocks.UpsampleConvSigmoid(32, out_channels),
                blocks.UpsampleConvSigmoid(16, out_channels),
                blocks.UpsampleConvSigmoid(8, out_channels),
                blocks.UpsampleConvSigmoid(8, out_channels)
            ]
        )
        self.output_conv = nn.Sequential(
            nn.Conv2d(out_channels * 5, out_channels, kernel_size=1),
            nn.Sigmoid()
        )

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, img):
        output_ecoder = []
        output_decoder = []
        result_encoder = []
        # encoder stage
        encoder_out = self.encoder[0](img)
        # encoder2_in index 0
        for i in range(4):
            output_ecoder.append(
                F.max_pool2d(encoder_out, kernel_size=2, stride=2)
            )
            encoder_out = self.encoder[1 + i](output_ecoder[i])
        # decoder stage
        output_decoder.append(
            self.decoder[0](encoder_out)
        )
        for i in range(4):
            decoder_out = utils.upsample_like(
                torch.cat(
                    (
                        output_decoder[i],
                        output_ecoder[3 - i]
                    ),
                    dim=1
                ),
                output_ecoder[2 - i] if i < 3 else img
            )
            output_decoder.append(
                self.decoder[i+1](decoder_out)
            )
        # output stage
        for i in range(5):
            result_encoder.append(
                self.decoder_resulting[i](
                    output_decoder[i], img.shape[2:]
                )
            )
        overall_result = self.output_conv(
            torch.cat(
                result_encoder,
                dim=1
            )
        )
        return overall_result
