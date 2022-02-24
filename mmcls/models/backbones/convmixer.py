import torch.nn as nn
from mmcv.cnn.bricks import build_activation_layer, build_norm_layer

from ..builder import BACKBONES
from ..utils import PatchEmbed
from .base_backbone import BaseBackbone


class Residual(nn.Module):

    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x


@BACKBONES.register_module()
class ConvMixer(BaseBackbone):
    """ConvMixer.                              .

    A PyTorch implementation of : `A ConvMixer for the 2020s
    <https://arxiv.org/pdf/2201.09792.pdf>`_

    Modified from the `official repo
    <https://github.com/locuslab/convmixer/blob/main/convmixer.py>`_
    and `timm
    <https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/convmixer.py>`_.

    Args:
        arch (str | dict): The model's architecture. If string, it should be
            one of architecture in ``ConvMixer.arch_settings``. And if dict, it
            should include the following two keys:

            - embed_dims (int): The dimensions of patch embedding.
            - depth (int): Number of repetitions of ConvMixer Layer.
            - patch_size (int): The patch size.
            - kernel_size (int): The kernel size of depthwise conv layers.

            Defaults to '768/32'.
        in_channels (int): Number of input image channels. Defaults to 3.
        patch_size (int): The size of one patch in the patch embed layer.
            Defaults to 7.
        norm_cfg (dict): The config dict for norm layers.
            Defaults to ``dict(type='BN')``.
        act_cfg (dict): The config dict for activation after each convolution.
            Defaults to ``dict(type='GELU')``.
        patch_cfg (dict): Configs of patch embeding. Defaults to an empty dict.
        init_cfg (dict, optional): Initialization config dict.
    """
    arch_settings = {
        '768/32': {
            'embed_dims': 768,
            'depth': 32,
            'patch_size': 7,
            'kernel_size': 7
        },
        '1024/20': {
            'embed_dims': 1024,
            'depth': 20,
            'patch_size': 14,
            'kernel_size': 9
        },
        '1536/20': {
            'embed_dims': 1536,
            'depth': 20,
            'patch_size': 7,
            'kernel_size': 9
        },
    }

    def __init__(self,
                 arch='small',
                 img_size=224,
                 in_channels=3,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='GELU'),
                 patch_cfg=dict(),
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)

        if isinstance(arch, str):
            assert arch in self.arch_settings, \
                f'Unavailable arch, please choose from ' \
                f'({set(self.arch_settings)}) or pass a dict.'
            arch = self.arch_settings[arch]
        elif isinstance(arch, dict):
            assert 'depths' in arch and 'channels' in arch, \
                f'The arch dict must have "depths" and "channels", ' \
                f'but got {list(arch.keys())}.'

        self.embed_dims = arch['embed_dims']
        self.depth = arch['depth']
        self.patch_size = arch['patch_size']
        self.kernel_size = arch['kernel_size']

        # Set patch embedding
        _patch_cfg = dict(
            img_size=img_size,
            in_channels=in_channels,
            embed_dims=self.embed_dims,
            conv_cfg=dict(
                type='Conv2d',
                kernel_size=self.patch_size,
                stride=self.patch_size),
        )
        _patch_cfg.update(patch_cfg)
        self.patch_embed = PatchEmbed(**_patch_cfg)
        self.act = build_activation_layer(act_cfg)
        self.norm = build_norm_layer(norm_cfg, self.embed_dims)[1]

        # Repetitions of ConvMixer Layer
        self.layers = nn.Sequential(*[
            nn.Sequential(
                Residual(
                    nn.Sequential(
                        nn.Conv2d(
                            self.embed_dims,
                            self.embed_dims,
                            self.kernel_size,
                            groups=self.embed_dims,
                            padding='same'), build_activation_layer(act_cfg),
                        build_norm_layer(norm_cfg, self.embed_dims)[1])),
                nn.Conv2d(self.embed_dims, self.embed_dims, kernel_size=1),
                build_activation_layer(act_cfg),
                build_norm_layer(norm_cfg, self.embed_dims)[1])
            for _ in range(self.depth)
        ])

    def forward(self, x):
        x = self.patch_embed(x)
        x = self.layers(x)
        gap = x.mean([-2, -1], keepdim=True)

        return (gap, )
