# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn.functional as F
from torch import nn

from ..builder import BACKBONES


class ResBlock(nn.Module):

    def __init__(self, embed_dims=16):
        super(ResBlock, self).__init__()

        self.conv1 = nn.Conv2d(
            embed_dims, embed_dims, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(embed_dims)
        self.leakyrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        self.conv2 = nn.Conv2d(
            embed_dims, embed_dims, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(embed_dims)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)

        out = self.leakyrelu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out += residual

        return out


@BACKBONES.register_module()
class LearnableResizer(nn.Module):
    """[Learning to Resize Images for Computer Vision Tasks](
    https://paperswithcode.com/paper/learning-to-resize-images-for-computer-
    vision)

    Parameters
    ----------
    in_channels (int): Number of input image channels. Default: 3.
    num_blocks (int): Number of res blocks. Default: 1.
    middle_channels (int): Feature size of hidden layer. Default: 16.
    output_size (tuple): output image size. Default: (224, 224).
    """

    def __init__(self,
                 in_channels=3,
                 num_blocks=1,
                 middle_channels=16,
                 output_size=(224, 224)):
        super().__init__()
        self.output_size = output_size
        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=middle_channels,
            kernel_size=7,
            stride=1,
            padding=3)
        self.leakyrelu1 = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        self.conv2 = nn.Conv2d(
            middle_channels, middle_channels, kernel_size=1, stride=1)
        self.leakyrelu2 = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.bn1 = nn.BatchNorm2d(middle_channels)

        self.resblock = self._make_block(num_blocks, middle_channels)

        self.conv3 = nn.Conv2d(
            middle_channels,
            middle_channels,
            kernel_size=3,
            stride=1,
            padding=1)
        self.bn2 = nn.BatchNorm2d(middle_channels)

        self.conv4 = nn.Conv2d(
            middle_channels,
            out_channels=in_channels,
            kernel_size=7,
            stride=1,
            padding=3)

    def _make_block(self, num_blocks, embed_dims):
        residual = []

        for i in range(num_blocks):
            block = ResBlock(embed_dims=embed_dims)
            residual.append(block)

        return nn.Sequential(*residual)

    def forward(self, x):
        residual = F.interpolate(x, size=self.output_size, mode='bilinear')

        out = self.conv1(x)
        out = self.leakyrelu1(out)

        out = self.conv2(out)
        out = self.leakyrelu2(out)
        out = self.bn1(out)

        out_residual = F.interpolate(
            out, size=self.output_size, mode='bilinear')

        out = self.resblock(out_residual)

        out = self.conv3(out)
        out = self.bn2(out)
        out += out_residual

        out = self.conv4(out)
        out += residual

        return out
