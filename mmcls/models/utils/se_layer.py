import torch.nn as nn


class SELayer(nn.Module):
    """Squeeze-and-Excitation Module.

    Args:
        inplanes (int): The input channels of the SEBottleneck block.
        ratio (int): Squeeze ratio in SELayer. Default: 16
    """

    def __init__(self, inplanes, ratio=16):
        super(SELayer, self).__init__()
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.conv1 = nn.Conv2d(
            inplanes, int(inplanes / ratio), kernel_size=1, stride=1)
        self.conv2 = nn.Conv2d(
            int(inplanes / ratio), inplanes, kernel_size=1, stride=1)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.global_avgpool(x)

        out = self.conv1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.sigmoid(out)
        return x * out
