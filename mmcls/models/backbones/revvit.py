# Copyright (c) OpenMMLab. All rights reserved.
import sys
from typing import Sequence

import numpy as np
import torch
from mmcv.cnn import build_norm_layer
from mmcv.cnn.bricks.drop import build_dropout
from mmcv.cnn.bricks.transformer import FFN, PatchEmbed
from mmengine.model import BaseModule, ModuleList
from torch import nn
from torch.autograd import Function as Function

from mmcls.models.backbones.base_backbone import BaseBackbone
from mmcls.registry import MODELS
from ..utils import MultiheadAttention, resize_pos_embed, to_2tuple


class RevBackProp(Function):
    """Custom Backpropagation function to allow (A) flushing memory in forward
    and (B) activation recomputation reversibly in backward for gradient
    calculation.

    Inspired by
    https://github.com/RobinBruegger/RevTorch/blob/master/revtorch/revtorch.py
    """

    @staticmethod
    def forward(
            ctx,
            x,
            layers,
            buffer_layers,  # List of layer ids for int activation to buffer
    ):
        """Reversible Forward pass.

        Any intermediate activations from `buffer_layers` are cached in ctx for
        forward pass. This is not necessary for standard usecases. Each
        reversible layer implements its own forward pass logic.
        """
        buffer_layers.sort()
        x1, x2 = torch.chunk(x, 2, dim=-1)
        intermediate = []

        for layer in layers:
            x1, x2 = layer(x1, x2)
            if layer.layer_id in buffer_layers:
                intermediate.extend([x1.detach(), x2.detach()])

        if len(buffer_layers) == 0:
            all_tensors = [x1.detach(), x2.detach()]
        else:
            intermediate = [torch.LongTensor(buffer_layers), *intermediate]
            all_tensors = [x1.detach(), x2.detach(), *intermediate]

        ctx.save_for_backward(*all_tensors)
        ctx.layers = layers

        return torch.cat([x1, x2], dim=-1)

    @staticmethod
    def backward(ctx, dx):
        """Reversible Backward pass.

        Any intermediate activations from `buffer_layers` are recovered from
        ctx. Each layer implements its own loic for backward pass (both
        activation recomputation and grad calculation).
        """
        d_x1, d_x2 = torch.chunk(dx, 2, dim=-1)
        # retrieve params from ctx for backward
        x1, x2, *int_tensors = ctx.saved_tensors
        # no buffering
        if len(int_tensors) != 0:
            buffer_layers = int_tensors[0].tolist()
        else:
            buffer_layers = []

        layers = ctx.layers

        for _, layer in enumerate(layers[::-1]):
            if layer.layer_id in buffer_layers:
                x1, x2, d_x1, d_x2 = layer.backward_pass(
                    y1=int_tensors[buffer_layers.index(layer.layer_id) * 2 +
                                   1],
                    y2=int_tensors[buffer_layers.index(layer.layer_id) * 2 +
                                   2],
                    d_y1=d_x1,
                    d_y2=d_x2,
                )
            else:
                x1, x2, d_x1, d_x2 = layer.backward_pass(
                    y1=x1,
                    y2=x2,
                    d_y1=d_x1,
                    d_y2=d_x2,
                )

        dx = torch.cat([d_x1, d_x2], dim=-1)

        del int_tensors
        del d_x1, d_x2, x1, x2

        return dx, None, None


class RevTransformerEncoderLayer(BaseModule):

    def __init__(self,
                 embed_dims,
                 num_heads,
                 feedforward_channels,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 num_fcs=2,
                 qkv_bias=True,
                 act_cfg=dict(type='GELU'),
                 norm_cfg=dict(type='LN'),
                 layer_id=0,
                 init_cfg=None):
        super(RevTransformerEncoderLayer, self).__init__(init_cfg=init_cfg)

        self.drop_path_cfg = dict(type='DropPath', drop_prob=drop_path_rate)
        self.embed_dims = embed_dims

        self.norm1_name, norm1 = build_norm_layer(
            norm_cfg, self.embed_dims, postfix=1)
        self.add_module(self.norm1_name, norm1)

        self.attn = MultiheadAttention(
            embed_dims=embed_dims,
            num_heads=num_heads,
            attn_drop=attn_drop_rate,
            proj_drop=drop_rate,
            qkv_bias=qkv_bias)

        self.norm2_name, norm2 = build_norm_layer(
            norm_cfg, self.embed_dims, postfix=2)
        self.add_module(self.norm2_name, norm2)

        self.ffn = FFN(
            embed_dims=embed_dims,
            feedforward_channels=feedforward_channels,
            num_fcs=num_fcs,
            ffn_drop=drop_rate,
            act_cfg=act_cfg,
            add_identity=False)

        self.layer_id = layer_id
        self.seeds = {}

    @property
    def norm1(self):
        return getattr(self, self.norm1_name)

    @property
    def norm2(self):
        return getattr(self, self.norm2_name)

    def init_weights(self):
        super(RevTransformerEncoderLayer, self).init_weights()
        for m in self.ffn.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.normal_(m.bias, std=1e-6)

    def seed_cuda(self, key):
        """Fix seeds to allow for stochastic elements such as dropout to be
        reproduced exactly in activation recomputation in the backward pass."""
        # randomize seeds
        # use cuda generator if available
        if (hasattr(torch.cuda, 'default_generators')
                and len(torch.cuda.default_generators) > 0):
            # GPU
            device_idx = torch.cuda.current_device()
            seed = torch.cuda.default_generators[device_idx].seed()
        else:
            # CPU
            seed = int(torch.seed() % sys.maxsize)

        self.seeds[key] = seed
        torch.manual_seed(self.seeds[key])

    def forward(self, x1, x2):
        """
        Implementation of Reversible TransformerEncoderLayer
        ```
        x = x + self.attn(self.norm1(x))
        x = self.ffn(self.norm2(x), identity=x)
        ```
        """
        self.seed_cuda('attn')
        # attention output
        f_x2 = self.attn(self.norm1(x2))
        # apply droppath on attention output
        self.seed_cuda('droppath')
        f_x2_dropped = build_dropout(self.drop_path_cfg)(f_x2)
        y1 = x1 + f_x2_dropped

        # free memory
        if self.training:
            del x1

        # ffn output
        self.seed_cuda('ffn')
        g_y1 = self.ffn(self.norm2(y1))
        # apply droppath on ffn output
        torch.manual_seed(self.seeds['droppath'])
        g_y1_dropped = build_dropout(self.drop_path_cfg)(g_y1)
        # final output
        y2 = x2 + g_y1_dropped

        # free memory
        if self.training:
            del x2

        return y1, y2

    def backward_pass(self, y1, y2, d_y1, d_y2):
        """equation for activation recomputation:

        X_2 = Y_2 - G(Y_1), G = FFN
        X_1 = Y_1 - F(X_2), F = MSHA
        """

        # temporarily record intermediate activation for G
        # and use them for gradient calculation of G
        with torch.enable_grad():
            y1.requires_grad = True

            torch.manual_seed(self.seeds['ffn'])
            g_y1 = self.ffn(self.norm2(y1))

            torch.manual_seed(self.seeds['droppath'])
            g_y1 = build_dropout(self.drop_path_cfg)(g_y1)

            g_y1.backward(d_y2, retain_graph=True)

        # activate recomputation is by design and not part of
        # the computation graph in forward pass
        with torch.no_grad():
            x2 = y2 - g_y1
            del g_y1

            d_y1 = d_y1 + y1.grad
            y1.grad = None

        # record F activation and calculate gradients on F
        with torch.enable_grad():
            x2.requires_grad = True

            torch.manual_seed(self.seeds['attn'])
            f_x2 = self.attn(self.norm1(x2))

            torch.manual_seed(self.seeds['droppath'])
            f_x2 = build_dropout(self.drop_path_cfg)(f_x2)

            f_x2.backward(d_y1, retain_graph=True)

        # propagate reverse computed activations at the
        # start of the previous block
        with torch.no_grad():
            x1 = y1 - f_x2
            del f_x2, y1

            d_y2 = d_y2 + x2.grad

            x2.grad = None
            x2 = x2.detach()

        return x1, x2, d_y1, d_y2


class TwoStreamFusion(nn.Module):

    def __init__(self, mode, dim=None, kernel=3, padding=1):
        """A general constructor for neural modules fusing two equal sized
        tensors in forward. Following options are supported:

        "add" / "max" / "min" / "avg"             :     respective operations
        on the two halves. "concat"                                  :
        NOOP. "concat_linear_{dim_mult}_{drop_rate}"    :     MLP to fuse with
        hidden dim "dim_mult"     (optional, def 1.) higher than input dim
        with optional dropout "drop_rate" (def: 0.)
        "ln+concat_linear_{dim_mult}_{drop_rate}" :     perform MLP after
        layernorm on the input.
        """
        super().__init__()
        self.mode = mode
        if mode == 'add':
            self.fuse_fn = lambda x: torch.stack(torch.chunk(x, 2, dim=2)).sum(
                dim=0)
        elif mode == 'max':
            self.fuse_fn = (lambda x: torch.stack(torch.chunk(x, 2, dim=2)).
                            max(dim=0).values)
        elif mode == 'min':
            self.fuse_fn = (lambda x: torch.stack(torch.chunk(x, 2, dim=2)).
                            min(dim=0).values)
        elif mode == 'avg':
            self.fuse_fn = lambda x: torch.stack(torch.chunk(x, 2, dim=2)
                                                 ).mean(dim=0)
        elif mode == 'concat':
            # x itself is the channel concat version
            self.fuse_fn = lambda x: x
        elif 'concat_linear' in mode:
            # TODO: add this style from Slow
            raise NotImplementedError

        else:
            raise NotImplementedError

    def forward(self, x):
        if 'concat_linear' in self.mode:
            return self.fuse_fn(x) + x

        else:
            return self.fuse_fn(x)


@MODELS.register_module()
class RevVisionTransformer(BaseBackbone):
    """Reversible Vision Transformer."""
    arch_zoo = {
        **dict.fromkeys(
            ['s', 'small'], {
                'embed_dims': 768,
                'num_layers': 8,
                'num_heads': 8,
                'feedforward_channels': 768 * 3,
            }),
        **dict.fromkeys(
            ['b', 'base'], {
                'embed_dims': 768,
                'num_layers': 12,
                'num_heads': 12,
                'feedforward_channels': 3072
            }),
        **dict.fromkeys(
            ['l', 'large'], {
                'embed_dims': 1024,
                'num_layers': 24,
                'num_heads': 16,
                'feedforward_channels': 4096
            }),
        **dict.fromkeys(
            ['h', 'huge'],
            {
                # The same as the implementation in MAE
                # <https://arxiv.org/abs/2111.06377>
                'embed_dims': 1280,
                'num_layers': 32,
                'num_heads': 16,
                'feedforward_channels': 5120
            }),
        **dict.fromkeys(
            ['deit-t', 'deit-tiny'], {
                'embed_dims': 192,
                'num_layers': 12,
                'num_heads': 3,
                'feedforward_channels': 192 * 4
            }),
        **dict.fromkeys(
            ['deit-s', 'deit-small'], {
                'embed_dims': 384,
                'num_layers': 12,
                'num_heads': 6,
                'feedforward_channels': 384 * 4
            }),
        **dict.fromkeys(
            ['deit-b', 'deit-base'], {
                'embed_dims': 768,
                'num_layers': 12,
                'num_heads': 12,
                'feedforward_channels': 768 * 4
            }),
    }
    # Some structures have multiple extra tokens, like DeiT.
    # RevViT does not allow cls_token
    num_extra_tokens = 1  # cls_token

    def __init__(
            self,
            arch='base',
            img_size=224,
            patch_size=16,
            in_channels=3,
            out_indices=-1,
            drop_rate=0.,
            drop_path_rate=0.,
            qkv_bias=True,
            norm_cfg=dict(type='LN', eps=1e-6),
            final_norm=True,
            with_cls_token=True,
            avg_token=False,
            frozen_stages=-1,
            output_cls_token=True,
            beit_style=False,
            #  layer_scale_init_value=0.1, # used for BEiT
            interpolate_mode='bicubic',
            patch_cfg=dict(),
            layer_cfgs=dict(),
            fusion_mode='concat',
            no_custom_backward=False,
            init_cfg=None):
        super(RevVisionTransformer, self).__init__(init_cfg)

        if isinstance(arch, str):
            arch = arch.lower()
            assert arch in set(self.arch_zoo), \
                f'Arch {arch} is not in default archs {set(self.arch_zoo)}'
            self.arch_settings = self.arch_zoo[arch]
        else:
            essential_keys = {
                'embed_dims', 'num_layers', 'num_heads', 'feedforward_channels'
            }
            assert isinstance(arch, dict) and essential_keys <= set(arch), \
                f'Custom arch needs a dict with keys {essential_keys}'
            self.arch_settings = arch

        self.embed_dims = self.arch_settings['embed_dims']
        self.num_layers = self.arch_settings['num_layers']
        self.img_size = to_2tuple(img_size)
        self.no_custom_backward = no_custom_backward

        # Set patch embedding
        _patch_cfg = dict(
            in_channels=in_channels,
            input_size=img_size,
            embed_dims=self.embed_dims,
            conv_type='Conv2d',
            kernel_size=patch_size,
            stride=patch_size,
        )
        _patch_cfg.update(patch_cfg)
        self.patch_embed = PatchEmbed(**_patch_cfg)
        self.patch_resolution = self.patch_embed.init_out_size
        num_patches = self.patch_resolution[0] * self.patch_resolution[1]

        # Set cls token
        if output_cls_token:
            assert with_cls_token is True, f'with_cls_token must be True if' \
                f'set output_cls_token to True, but got {with_cls_token}'
        self.with_cls_token = with_cls_token
        self.output_cls_token = output_cls_token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dims))

        # Set position embedding
        self.interpolate_mode = interpolate_mode
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + self.num_extra_tokens,
                        self.embed_dims))
        self._register_load_state_dict_pre_hook(self._prepare_pos_embed)

        self.drop_after_pos = nn.Dropout(p=drop_rate)

        if isinstance(out_indices, int):
            out_indices = [out_indices]
        assert isinstance(out_indices, Sequence), \
            f'"out_indices" must by a sequence or int, ' \
            f'get {type(out_indices)} instead.'
        for i, index in enumerate(out_indices):
            if index < 0:
                out_indices[i] = self.num_layers + index
            assert 0 <= out_indices[i] <= self.num_layers, \
                f'Invalid out_indices {index}'
        self.out_indices = out_indices

        # stochastic depth decay rule
        dpr = np.linspace(0, drop_path_rate, self.num_layers)

        self.layers = ModuleList()
        if isinstance(layer_cfgs, dict):
            layer_cfgs = [layer_cfgs] * self.num_layers
        for i in range(self.num_layers):
            _layer_cfg = dict(
                embed_dims=self.embed_dims,
                num_heads=self.arch_settings['num_heads'],
                feedforward_channels=self.
                arch_settings['feedforward_channels'],
                drop_rate=drop_rate,
                drop_path_rate=dpr[i],
                qkv_bias=qkv_bias,
                layer_id=i,
                norm_cfg=norm_cfg)
            _layer_cfg.update(layer_cfgs[i])
            if beit_style:
                raise NotImplementedError
            else:
                self.layers.append(RevTransformerEncoderLayer(**_layer_cfg))

        self.fusion_layer = TwoStreamFusion(
            mode=fusion_mode, dim=self.embed_dims)

        self.frozen_stages = frozen_stages
        self.final_norm = final_norm
        if final_norm:
            self.norm1_name, norm1 = build_norm_layer(
                norm_cfg, self.embed_dims * 2, postfix=1)
            self.add_module(self.norm1_name, norm1)

        self.avg_token = avg_token
        # if avg_token:
        #     self.norm2_name, norm2 = build_norm_layer(
        #         norm_cfg, self.embed_dims, postfix=2)
        #     self.add_module(self.norm2_name, norm2)
        # freeze stages only when self.frozen_stages > 0
        if self.frozen_stages > 0:
            self._freeze_stages()

    @property
    def norm1(self):
        return getattr(self, self.norm1_name)

    # @property
    # def norm2(self):
    #     return getattr(self, self.norm2_name)

    def _prepare_pos_embed(self, state_dict, prefix, *args, **kwargs):
        name = prefix + 'pos_embed'
        if name not in state_dict.keys():
            return

        ckpt_pos_embed_shape = state_dict[name].shape
        if self.pos_embed.shape != ckpt_pos_embed_shape:
            from mmengine.logging import MMLogger
            logger = MMLogger.get_current_instance()
            logger.info(
                f'Resize the pos_embed shape from {ckpt_pos_embed_shape} '
                f'to {self.pos_embed.shape}.')

            ckpt_pos_embed_shape = to_2tuple(
                int(np.sqrt(ckpt_pos_embed_shape[1] - self.num_extra_tokens)))
            pos_embed_shape = self.patch_embed.init_out_size

            state_dict[name] = resize_pos_embed(state_dict[name],
                                                ckpt_pos_embed_shape,
                                                pos_embed_shape,
                                                self.interpolate_mode,
                                                self.num_extra_tokens)

    @staticmethod
    def resize_pos_embed(*args, **kwargs):
        """Interface for backward-compatibility."""
        return resize_pos_embed(*args, **kwargs)

    def _freeze_stages(self):
        # freeze position embedding
        self.pos_embed.requires_grad = False
        # set dropout to eval model
        self.drop_after_pos.eval()
        # freeze patch embedding
        self.patch_embed.eval()
        for param in self.patch_embed.parameters():
            param.requires_grad = False
        # freeze cls_token
        # self.cls_token.requires_grad = False
        # freeze layers
        for i in range(1, self.frozen_stages + 1):
            m = self.layers[i - 1]
            m.eval()
            for param in m.parameters():
                param.requires_grad = False
        # freeze the last layer norm
        if self.frozen_stages == len(self.layers) and self.final_norm:
            self.norm1.eval()
            for param in self.norm1.parameters():
                param.requires_grad = False

    def forward(self, x):
        B = x.shape[0]
        x, patch_resolution = self.patch_embed(x)

        # stole cls_tokens impl from Phil Wang, thanks
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        x = x + resize_pos_embed(
            self.pos_embed,
            self.patch_resolution,
            patch_resolution,
            mode=self.interpolate_mode,
            num_extra_tokens=self.num_extra_tokens)
        # x = x + self.pos_embed
        x = self.drop_after_pos(x)

        if not self.with_cls_token:
            # Remove class token for transformer encoder input
            x = x[:, 1:]

        x = torch.cat([x, x], dim=-1)

        # forward with different conditions
        if not self.training or self.no_custom_backward:
            # in eval/inference model
            executing_fn = RevVisionTransformer._forward_vanilla_bp
        else:
            # use custom backward when self.training=True.
            executing_fn = RevBackProp.apply

        x = executing_fn(x, self.layers, [])

        if self.final_norm:
            x = self.norm1(x)
        x = self.fusion_layer(x)

        if self.with_cls_token:
            # RevViT does not allow cls_token
            raise NotImplementedError
        else:
            # (B, H, W, C)
            _, __, C = x.shape
            patch_token = x.reshape(B, *patch_resolution, C)
            # (B, C, H, W)
            patch_token = patch_token.permute(0, 3, 1, 2)
            cls_token = None

        if self.avg_token:
            # (B, H, W, C)
            patch_token = patch_token.permute(0, 2, 3, 1)
            # (B, L, C) -> (B, C)
            patch_token = patch_token.reshape(
                B, patch_resolution[0] * patch_resolution[1], C).mean(dim=1)
            # patch_token = self.norm2(patch_token)

        if self.output_cls_token:
            out = [patch_token, cls_token]
        else:
            out = patch_token

        # # >>>>>>>>>>>>>>>>>>
        # outs = []
        # for i, layer in enumerate(self.layers):
        #     x = layer(x)

        #     if i == len(self.layers) - 1 and self.final_norm:
        #         x = self.norm1(x)

        #     if i in self.out_indices:
        #         B, _, C = x.shape
        #         if self.with_cls_token:
        #             # RevViT does not allow cls_token
        #             raise NotImplementedError
        #         else:
        #             patch_token = x.reshape(B, *patch_resolution, C)
        #             patch_token = patch_token.permute(0, 3, 1, 2)
        #             cls_token = None
        #         if self.avg_token:
        #             patch_token = patch_token.permute(0, 2, 3, 1)
        #             patch_token = patch_token.reshape(
        #                 B, patch_resolution[0] * patch_resolution[1],
        #                 C).mean(dim=1)
        #             patch_token = self.norm2(patch_token)
        #         if self.output_cls_token:
        #             out = [patch_token, cls_token]
        #         else:
        #             out = patch_token
        #         outs.append(out)

        return tuple([out])

    @staticmethod
    def _forward_vanilla_bp(hidden_state, layers, buffer=[]):
        """Using reversible layers without reversible backpropagation.

        Debugging purpose only. Activated with self.no_custom_backward
        """
        # split into ffn state(ffn_out) and attention output(attn_out)
        ffn_out, attn_out = torch.chunk(hidden_state, 2, dim=-1)
        del hidden_state

        for i, layer in enumerate(layers):
            attn_out, ffn_out = layer(attn_out, ffn_out)

        return torch.cat([attn_out, ffn_out], dim=-1)
