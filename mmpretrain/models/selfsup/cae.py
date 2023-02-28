# Copyright (c) OpenMMLab. All rights reserved.
# Part of code is modified from BEiT
# https://github.com/microsoft/unilm/blob/master/beit/dall_e/encoder.py
import math
from collections import OrderedDict
from functools import partial
from typing import Dict, List, Optional, Union

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
from mmpretrain.structures import DataSample
from ..utils import build_2d_sincos_position_embedding
from .base import BaseSelfSupervisor


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
                    built for {self.input_channels}'                                                    )
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

    def forward(self, img: torch.Tensor,
                mask: Optional[torch.Tensor]) -> torch.Tensor:
        """Generate features for masked images.

        This function generates mask images and get the hidden features for
        visible patches.

        The function supports two kind of forward behaviors. If the ``mask`` is
        not ``None``, the forward function will be executed as masked image
        modeling pre-training; if the ``mask`` is ``None``, the forward
        function will call ``super().forward()``, which extract features from
        images without mask.

        Args:
            x (torch.Tensor): Input images, which is of shape B x C x H x W.
            mask (torch.Tensor, optional): Mask for input, which is of shape
                B x L.

        Returns:
            torch.Tensor: hidden features.
        """
        if mask is None:
            return super().forward(x)

        else:
            x, _ = self.patch_embed(img)
            batch_size, _, dim = x.size()

            cls_tokens = self.cls_token.expand(batch_size, -1, -1)

            # NOTE: unmasked embeddings
            x_unmasked = x[~mask].reshape(batch_size, -1, dim)
            x_unmasked = torch.cat((cls_tokens, x_unmasked), dim=1)

            pos_embed = self.pos_embed.expand(batch_size, self.num_patches + 1,
                                              dim)
            pos_embed_unmasked = pos_embed[:, 1:][~mask].reshape(
                batch_size, -1, dim)
            pos_embed_unmasked = torch.cat(
                (pos_embed[:, :1], pos_embed_unmasked), dim=1)
            x_unmasked = x_unmasked + pos_embed_unmasked

            x_unmasked = self.drop_after_pos(x_unmasked)

            for i, layer in enumerate(self.layers):
                x_unmasked = layer(x=x_unmasked, rel_pos_bias=None)

                if i == len(self.layers) - 1 and self.final_norm:
                    x_unmasked = self.norm1(x_unmasked)

            return x_unmasked


@MODELS.register_module()
class CAE(BaseSelfSupervisor):
    """CAE.

    Implementation of `Context Autoencoder for Self-Supervised Representation
    Learning <https://arxiv.org/abs/2202.03026>`_.

    Args:
        backbone (dict): Config dict for module of backbone.
        neck (dict): Config dict for module of neck.
        head (dict): Config dict for module of head functions.
        target_generator: (dict, optional): The target_generator module to
            generate targets for self-supervised learning optimization, such as
            HOG, extracted features from other modules(DALL-E, CLIP), etc.
        base_momentum (float): The base momentum coefficient for the target
            network. Defaults to 0.0.
        data_preprocessor (dict, optional): The config for preprocessing
            input data. If None or no specified type, it will use
            "SelfSupDataPreprocessor" as type.
            See :class:`SelfSupDataPreprocessor` for more details.
            Defaults to None.
        init_cfg (Union[List[dict], dict], optional): Config dict for weight
            initialization. Defaults to None.
    """

    def __init__(self,
                 backbone: dict,
                 neck: dict,
                 head: dict,
                 target_generator: Optional[dict] = None,
                 base_momentum: float = 0.0,
                 data_preprocessor: Optional[dict] = None,
                 init_cfg: Optional[Union[List[dict], dict]] = None) -> None:
        super().__init__(
            backbone=backbone,
            neck=neck,
            head=head,
            target_generator=target_generator,
            data_preprocessor=data_preprocessor,
            init_cfg=init_cfg)

        self.momentum = base_momentum
        self.teacher = MODELS.build(backbone)

    def init_weights(self) -> None:
        """Initialize weights."""
        super().init_weights()
        self._init_teacher()

    def _init_teacher(self) -> None:
        """Init the weights of teacher with those of backbone."""
        for param_backbone, param_teacher in zip(self.backbone.parameters(),
                                                 self.teacher.parameters()):
            param_teacher.detach()
            param_teacher.data.copy_(param_backbone.data)
            param_teacher.requires_grad = False

    def momentum_update(self) -> None:
        """Momentum update of the teacher network."""
        for param_bacbone, param_teacher in zip(self.backbone.parameters(),
                                                self.teacher.parameters()):
            param_teacher.data = param_teacher.data * self.momentum + \
                param_bacbone.data * (1. - self.momentum)

    def loss(self, inputs: List[torch.Tensor], data_samples: List[DataSample],
             **kwargs) -> Dict[str, torch.Tensor]:
        """The forward function in training.

        Args:
            inputs (List[torch.Tensor]): The input images.
            data_samples (List[SelfSupDataSample]): All elements required
                during the forward function.

        Returns:
            Dict[str, torch.Tensor]: A dictionary of loss components.
        """
        mask = torch.stack([data_sample.mask for data_sample in data_samples])
        mask = mask.flatten(1).to(torch.bool)

        unmasked = self.backbone(inputs[0], mask)

        # get the latent prediction for the masked patches
        with torch.no_grad():
            # inputs[0] is the prediction image
            latent_target = self.teacher(inputs[0], ~mask)
            latent_target = latent_target[:, 1:, :]
            self.momentum_update()

        pos_embed = self.backbone.pos_embed.expand(inputs[0].shape[0], -1, -1)
        pos_embed_masked = pos_embed[:,
                                     1:][mask].reshape(inputs[0].shape[0], -1,
                                                       pos_embed.shape[-1])
        pos_embed_unmasked = pos_embed[:, 1:][~mask].reshape(
            inputs[0].shape[0], -1, pos_embed.shape[-1])

        # input the unmasked tokens and masked tokens to the decoder
        logits, latent_pred = self.neck(unmasked[:, 1:], pos_embed_masked,
                                        pos_embed_unmasked)

        logits = logits.view(-1, logits.shape[-1])
        # inputs[1] is the target image
        logits_target = self.target_generator(inputs[1])
        loss_main, loss_align = self.head(logits, logits_target, latent_pred,
                                          latent_target, mask)
        losses = dict()

        losses['loss'] = loss_main + loss_align
        losses['main'] = loss_main
        losses['align'] = loss_align
        return losses
