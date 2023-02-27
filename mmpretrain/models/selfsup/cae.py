# Copyright (c) OpenMMLab. All rights reserved.
# Part of code is modified from BEiT
# https://github.com/microsoft/unilm/blob/master/beit/dall_e/encoder.py
import math
from collections import OrderedDict
from functools import partial
from typing import List, Optional, Union

import attr
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.model import BaseModule, ModuleList
from mmengine.model.weight_init import trunc_normal_

from mmpretrain.models import VisionTransformer
from mmpretrain.models.backbones.beit import BEiTTransformerEncoderLayer
from mmpretrain.registry import MODELS
from ..utils import build_2d_sincos_position_embedding


@attr.s(eq=False)
class Conv2d(nn.Module):
    n_in: int = attr.ib(validator=lambda i, a, x: x >= 1)
    n_out: int = attr.ib(validator=lambda i, a, x: x >= 1)
    kw: int = attr.ib(validator=lambda i, a, x: x >= 1 and x % 2 == 1)

    use_float16: bool = attr.ib(default=True)
    device: torch.device = attr.ib(default=torch.device('cpu'))
    requires_grad: bool = attr.ib(default=False)

    def __attrs_post_init__(self) -> None:
        super().__init__()

        w = torch.empty((self.n_out, self.n_in, self.kw, self.kw),
                        dtype=torch.float32,
                        device=self.device,
                        requires_grad=self.requires_grad)
        w.normal_(std=1 / math.sqrt(self.n_in * self.kw**2))

        b = torch.zeros((self.n_out, ),
                        dtype=torch.float32,
                        device=self.device,
                        requires_grad=self.requires_grad)
        self.w, self.b = nn.Parameter(w), nn.Parameter(b)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_float16 and 'cuda' in self.w.device.type:
            if x.dtype != torch.float16:
                x = x.half()

            w, b = self.w.half(), self.b.half()
        else:
            if x.dtype != torch.float32:
                x = x.float()

            w, b = self.w, self.b

        return F.conv2d(x, w, b, padding=(self.kw - 1) // 2)


@attr.s(eq=False, repr=False)
class EncoderBlock(nn.Module):
    n_in: int = attr.ib(validator=lambda i, a, x: x >= 1)
    n_out: int = attr.ib(validator=lambda i, a, x: x >= 1 and x % 4 == 0)
    n_layers: int = attr.ib(validator=lambda i, a, x: x >= 1)

    device: torch.device = attr.ib(default=None)
    requires_grad: bool = attr.ib(default=False)

    def __attrs_post_init__(self) -> None:
        super().__init__()
        self.n_hid = self.n_out // 4
        self.post_gain = 1 / (self.n_layers**2)

        make_conv = partial(
            Conv2d, device=self.device, requires_grad=self.requires_grad)
        self.id_path = make_conv(
            self.n_in, self.n_out,
            1) if self.n_in != self.n_out else nn.Identity()
        self.res_path = nn.Sequential(
            OrderedDict([
                ('relu_1', nn.ReLU()),
                ('conv_1', make_conv(self.n_in, self.n_hid, 3)),
                ('relu_2', nn.ReLU()),
                ('conv_2', make_conv(self.n_hid, self.n_hid, 3)),
                ('relu_3', nn.ReLU()),
                ('conv_3', make_conv(self.n_hid, self.n_hid, 3)),
                ('relu_4', nn.ReLU()),
                ('conv_4', make_conv(self.n_hid, self.n_out, 1)),
            ]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.id_path(x) + self.post_gain * self.res_path(x)


@attr.s(eq=False, repr=False)
@MODELS.register_module(name='DALL-E')
class Encoder(BaseModule):
    group_count: int = 4
    n_hid: int = attr.ib(default=256, validator=lambda i, a, x: x >= 64)
    n_blk_per_group: int = attr.ib(default=2, validator=lambda i, a, x: x >= 1)
    input_channels: int = attr.ib(default=3, validator=lambda i, a, x: x >= 1)
    vocab_size: int = attr.ib(default=8192, validator=lambda i, a, x: x >= 512)

    device: torch.device = attr.ib(default=torch.device('cpu'))
    requires_grad: bool = attr.ib(default=False)
    use_mixed_precision: bool = attr.ib(default=True)
    init_cfg: Optional[Union[dict, List[dict]]] = attr.ib(default=None)

    def __attrs_post_init__(self) -> None:
        super().__init__(init_cfg=self.init_cfg)

        blk_range = range(self.n_blk_per_group)
        n_layers = self.group_count * self.n_blk_per_group
        make_conv = partial(
            Conv2d, device=self.device, requires_grad=self.requires_grad)
        make_blk = partial(
            EncoderBlock,
            n_layers=n_layers,
            device=self.device,
            requires_grad=self.requires_grad)

        self.blocks = nn.Sequential(
            OrderedDict([
                ('input', make_conv(self.input_channels, 1 * self.n_hid, 7)),
                ('group_1',
                 nn.Sequential(
                     OrderedDict([
                         *[(f'block_{i + 1}',
                            make_blk(1 * self.n_hid, 1 * self.n_hid))
                           for i in blk_range],
                         ('pool', nn.MaxPool2d(kernel_size=2)),
                     ]))),
                ('group_2',
                 nn.Sequential(
                     OrderedDict([
                         *[(f'block_{i + 1}',
                            make_blk(
                                1 * self.n_hid if i == 0 else 2 * self.n_hid,
                                2 * self.n_hid)) for i in blk_range],
                         ('pool', nn.MaxPool2d(kernel_size=2)),
                     ]))),
                ('group_3',
                 nn.Sequential(
                     OrderedDict([
                         *[(f'block_{i + 1}',
                            make_blk(
                                2 * self.n_hid if i == 0 else 4 * self.n_hid,
                                4 * self.n_hid)) for i in blk_range],
                         ('pool', nn.MaxPool2d(kernel_size=2)),
                     ]))),
                ('group_4',
                 nn.Sequential(
                     OrderedDict([
                         *[(f'block_{i + 1}',
                            make_blk(
                                4 * self.n_hid if i == 0 else 8 * self.n_hid,
                                8 * self.n_hid)) for i in blk_range],
                     ]))),
                ('output',
                 nn.Sequential(
                     OrderedDict([
                         ('relu', nn.ReLU()),
                         ('conv',
                          make_conv(
                              8 * self.n_hid,
                              self.vocab_size,
                              1,
                              use_float16=False)),
                     ]))),
            ]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.float()
        if len(x.shape) != 4:
            raise ValueError(f'input shape {x.shape} is not 4d')
        if x.shape[1] != self.input_channels:
            raise ValueError(f'input has {x.shape[1]} channels but model \
                    built for {self.input_channels}')
        if x.dtype != torch.float32:
            raise ValueError('input must have dtype torch.float32')

        return self.blocks(x)


@MODELS.register_module()
class CAEViT(VisionTransformer):
    """Vision Transformer for CAE pre-training.

    Rewritten version of: `An Image is Worth 16x16 Words: Transformers
    for Image Recognition at Scale <https://arxiv.org/abs/2010.11929>`_

    Args:
        arch (str | dict): Vision Transformer architecture. Default: 'b'
        img_size (int | tuple): Input image size
        patch_size (int | tuple): The patch size
        out_indices (Sequence | int): Output from which stages.
            Defaults to -1, means the last stage.
        drop_rate (float): Probability of an element to be zeroed.
            Defaults to 0.
        drop_path_rate (float): stochastic depth rate. Defaults to 0.
        bias (bool | str): The option to add leanable bias for q, k, v. If bias
            is True, it will add leanable bias. If bias is 'qv_bias', it will
            only add leanable bias for q, v. If bias is False, it will not add
            bias for q, k, v. Default to 'qv_bias'.
        norm_cfg (dict): Config dict for normalization layer.
            Defaults to ``dict(type='LN')``.
        final_norm (bool): Whether to add a additional layer to normalize
            final feature map. Defaults to True.
        output_cls_token (bool): Whether output the cls_token. If set True,
            `with_cls_token` must be True. Defaults to True.
        interpolate_mode (str): Select the interpolate mode for position
            embeding vector resize. Defaults to "bicubic".
        layer_scale_init_value (float, optional): The init value of gamma in
            BEiTTransformerEncoderLayer.
        patch_cfg (dict): Configs of patch embeding. Defaults to an empty dict.
        layer_cfgs (Sequence | dict): Configs of each transformer layer in
            encoder. Defaults to an empty dict.
        init_cfg (dict, optional): Initialization config dict.
            Defaults to None.
    """

    def __init__(self,
                 arch: str = 'b',
                 img_size: int = 224,
                 patch_size: int = 16,
                 out_indices: int = -1,
                 drop_rate: float = 0,
                 drop_path_rate: float = 0,
                 bias: bool = 'qv_bias',
                 norm_cfg: dict = dict(type='LN', eps=1e-6),
                 final_norm: bool = True,
                 output_cls_token: bool = True,
                 interpolate_mode: str = 'bicubic',
                 layer_scale_init_value: float = None,
                 patch_cfg: dict = dict(),
                 layer_cfgs: dict = dict(),
                 init_cfg: dict = None) -> None:
        super().__init__(
            arch=arch,
            img_size=img_size,
            patch_size=patch_size,
            out_indices=out_indices,
            drop_rate=drop_rate,
            drop_path_rate=drop_path_rate,
            norm_cfg=norm_cfg,
            final_norm=final_norm,
            output_cls_token=output_cls_token,
            interpolate_mode=interpolate_mode,
            patch_cfg=patch_cfg,
            layer_cfgs=layer_cfgs,
            init_cfg=init_cfg)
        self.pos_embed.requires_grad = False
        self.num_patches = self.patch_resolution[0] * self.patch_resolution[1]
        dpr = np.linspace(0, drop_path_rate, self.num_layers)

        # Replace original TransformerEncoderLayer with
        # BEiTTransformerEncoderLayer
        self.layers = ModuleList()
        if isinstance(layer_cfgs, dict):
            layer_cfgs = [layer_cfgs] * self.num_layers
        for i in range(self.num_layers):
            _layer_cfg = dict(
                embed_dims=self.embed_dims,
                num_heads=self.arch_settings['num_heads'],
                feedforward_channels=self.
                arch_settings['feedforward_channels'],
                layer_scale_init_value=layer_scale_init_value,
                window_size=None,
                # setting `use_rel_pos_bias` to False ignores the `window_size`
                use_rel_pos_bias=False,
                drop_rate=drop_rate,
                drop_path_rate=dpr[i],
                bias=bias,
                norm_cfg=norm_cfg)
            _layer_cfg.update(layer_cfgs[i])
            self.layers.append(BEiTTransformerEncoderLayer(**_layer_cfg))

    def init_weights(self) -> None:
        """Initialize position embedding, patch embedding and cls token."""
        super().init_weights()
        if not (isinstance(self.init_cfg, dict)
                and self.init_cfg['type'] == 'Pretrained'):
            # initialize position  embedding in backbone
            pos_embed = build_2d_sincos_position_embedding(
                int(self.num_patches**.5),
                self.pos_embed.shape[-1],
                cls_token=True)
            self.pos_embed.data.copy_(pos_embed.float())

            trunc_normal_(self.cls_token, std=.02)
            self.apply(self._init_weights)

    def _init_weights(self, m) -> None:
        """Initialize the weights."""
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, img: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Generate features for masked images.

        This function generates mask images and get the hidden features for
        visible patches.

        Args:
            x (torch.Tensor): Input images, which is of shape B x C x H x W.
            mask (torch.Tensor): Mask for input, which is of shape B x L.

        Returns:
            torch.Tensor: hidden features.
        """
        x, _ = self.patch_embed(img)
        batch_size, _, dim = x.size()

        cls_tokens = self.cls_token.expand(batch_size, -1, -1)

        # NOTE: unmasked embeddings
        x_unmasked = x[~mask].reshape(batch_size, -1, dim)
        x_unmasked = torch.cat((cls_tokens, x_unmasked), dim=1)

        pos_embed = self.pos_embed.expand(batch_size, self.num_patches + 1,
                                          dim)
        pos_embed_unmasked = pos_embed[:,
                                       1:][~mask].reshape(batch_size, -1, dim)
        pos_embed_unmasked = torch.cat((pos_embed[:, :1], pos_embed_unmasked),
                                       dim=1)
        x_unmasked = x_unmasked + pos_embed_unmasked

        x_unmasked = self.drop_after_pos(x_unmasked)

        for i, layer in enumerate(self.layers):
            x_unmasked = layer(x=x_unmasked, rel_pos_bias=None)

            if i == len(self.layers) - 1 and self.final_norm:
                x_unmasked = self.norm1(x_unmasked)

        return x_unmasked
