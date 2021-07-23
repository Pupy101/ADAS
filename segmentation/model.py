import torch
import u2net_blocks as blocks
import utils

from torch import nn
from torch.nn import functional as F


class U2Net(nn.Module):
    """
    Implementation from https://arxiv.org/pdf/2005.09007.pdf
    U2net for multiple class or one class segmentation.
    Number of class can be changed by parametr \'out_channels\'.
    This NN equal U2Net from paper, but I drop one block in center of net - En_6.
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()

        # encoder
        self.encoder1 = blocks.RSU7OneDilation([in_channels, 16, 8, 16], 3, 1, 2, 2)
        self.encoder2 = blocks.RSU6OneDilation([16, 32, 16, 32], 3, 1, 2, 2)
        self.encoder3 = blocks.RSU5OneDilation([32, 64, 32, 64], 3, 1, 2, 2)
        self.encoder4 = blocks.RSU4OneDilation([64, 128, 64, 128], 3, 1, 2, 2)
        self.encoder5 = blocks.RSU4FiveDilation([128, 256, 128, 256], 3, 1, [2, 4, 8, 4, 2], [2, 4, 8, 4, 2])

        # decoder
        self.decoder1 = blocks.RSU4FiveDilation([256, 128, 64, 128], 3, 1, [2, 4, 8, 4, 2], [2, 4, 8, 4, 2])
        self.decoder2 = blocks.RSU4OneDilation([128 * 2, 64, 32, 64], 3, 1, 2, 2)
        self.decoder3 = blocks.RSU5OneDilation([64 * 2, 32, 16, 32], 3, 1, 2, 2)
        self.decoder4 = blocks.RSU6OneDilation([32 * 2, 16, 16, 16], 3, 1, 2, 2)
        self.decoder5 = blocks.RSU7OneDilation([16 * 2, 8, 8, 8], 3, 1, 2, 2)

        self.decoder1_resulting = blocks.UpsampleConvSigmoid(128, out_channels)
        self.decoder2_resulting = blocks.UpsampleConvSigmoid(64, out_channels)
        self.decoder3_resulting = blocks.UpsampleConvSigmoid(32, out_channels)
        self.decoder4_resulting = blocks.UpsampleConvSigmoid(16, out_channels)
        self.decoder5_resulting = blocks.UpsampleConvSigmoid(8, out_channels)

        self.output_conv = nn.Sequential(
            nn.Conv2d(out_channels * 5, out_channels, kernel_size=1),
            nn.Sigmoid()
        )

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, img):

        # encoder stage
        encoder1_out = self.encoder1(img)

        encoder2_in = F.max_pool2d(encoder1_out, kernel_size=2, stride=2)
        encoder2_out = self.encoder2(encoder2_in)

        encoder3_in = F.max_pool2d(encoder2_out, kernel_size=2, stride=2)
        encoder3_out = self.encoder3(encoder3_in)

        encoder4_in = F.max_pool2d(encoder3_out, kernel_size=2, stride=2)
        encoder4_out = self.encoder4(encoder4_in)

        encoder5_in = F.max_pool2d(encoder4_out, kernel_size=2, stride=2)
        encoder5_out = self.encoder5(encoder5_in)

        # decoder stage
        decoder1_out = self.decoder1(encoder5_out)

        decoder2_in = utils.UpsampleX2(torch.cat((decoder1_out, encoder5_in), dim=1))
        decoder2_out = self.decoder2(decoder2_in)

        decoder3_in = utils.UpsampleX2(torch.cat((decoder2_out, encoder4_in), dim=1))
        decoder3_out = self.decoder3(decoder3_in)

        decoder4_in = utils.UpsampleX2(torch.cat((decoder3_out, encoder3_in), dim=1))
        decoder4_out = self.decoder4(decoder4_in)

        decoder5_in = utils.UpsampleX2(torch.cat((decoder4_out, encoder2_in), dim=1))
        decoder5_out = self.decoder5(decoder5_in)

        # output stage
        result_encoder1 = self.decoder1_resulting(decoder1_out, img.shape[2:])
        result_encoder2 = self.decoder2_resulting(decoder2_out, img.shape[2:])
        result_encoder3 = self.decoder3_resulting(decoder3_out, img.shape[2:])
        result_encoder4 = self.decoder4_resulting(decoder4_out, img.shape[2:])
        result_encoder5 = self.decoder5_resulting(decoder5_out, img.shape[2:])

        overall_result = self.output_conv(
            torch.cat((result_encoder1, result_encoder2, result_encoder3, result_encoder4, result_encoder5), dim=1)
        )
        return overall_result
