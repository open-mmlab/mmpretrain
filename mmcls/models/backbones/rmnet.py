import torch.nn as nn
from ..builder import BACKBONES
from .base_backbone import BaseBackbone
@BACKBONES.register_module()
class RMNet(BaseBackbone):
    def __init__(self, depth,frozen_stages=-1):
        super(RMNet, self).__init__()
        self.frozen_stages = frozen_stages
        self.conv1 = nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.inplanes = 64
        stages = [2,2,2,2] if depth == 18 else [3,4,6,3]
        self.layer1 = self._make_layer(64, stages[0], 1)
        self.layer2 = self._make_layer(128, stages[1], 2)
        self.layer3 = self._make_layer(256, stages[2], 2)
        self.layer4 = self._make_layer(512, stages[3], 2)
        self._freeze_stages()
        
    def _make_layer(self, planes, blocks, stride=1):
        layers = []
        layers.append(
            nn.Sequential(
                nn.Conv2d(self.inplanes, self.inplanes+planes, kernel_size=3,stride=stride, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.inplanes+planes, planes, kernel_size=3,stride=1, padding=1),
                nn.ReLU(inplace=True)
            )
        )
        self.inplanes = planes
        for _ in range(1, blocks):
            layers.append(
                nn.Sequential(
                    nn.Conv2d(self.inplanes, self.inplanes+planes, kernel_size=3,stride=1, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(self.inplanes+planes, planes, kernel_size=3,stride=1, padding=1),
                    nn.ReLU(inplace=True)
                )
            )
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        outs = []
        x = self.layer1(x)
        outs.append(x)
        x = self.layer2(x)
        outs.append(x)
        x = self.layer3(x)
        outs.append(x)
        x = self.layer4(x)
        outs.append(x)
        return tuple(outs)
    
    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            self.bn1.eval()
            for m in [self.conv1, self.bn1]:
                for param in m.parameters():
                    param.requires_grad = False

        for i in range(1, self.frozen_stages + 1):
            m = getattr(self, f'layer{i}')
            m.eval()
            for param in m.parameters():
                param.requires_grad = False

    def train(self, mode=True):
        super(RMNet, self).train(mode)
        self._freeze_stages()
        if mode and self.norm_eval:
            for m in self.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()
