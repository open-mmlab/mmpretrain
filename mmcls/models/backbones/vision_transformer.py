from itertools import repeat

import torch
import torch.nn as nn
from mmcv.cnn import build_conv_layer, build_norm_layer

from ..builder import BACKBONES
from ..utils import FFN
from .base_backbone import BaseBackbone

# import torch.utils.checkpoint as cp
# from mmcv.cnn import (ConvModule, build_conv_layer, build_norm_layer,
#   constant_init, kaiming_init)


# TODO:
# MultiheadAttention: attention mask?
class MultiheadAttention(nn.Module):
    """MultiheadAttention module.

    The main difference between nn.MultiheadAttention and ours is that the
    torch implementation doesn't have drop-out after the final linear layer,
    which is needed by VisionTransformer.

    Args:
        embed_dims (int): The embedding dimensions, i.e. 'd_model' mentioned
            in the paper.
        num_heads (int): Number of head. Default: 8.
        qkv_bias (bool): Whether the qkv projections have bias.
            Default: False.
        attn_drop (float): The drop-out rate after the attention operation.
            Default: 0.0.
        proj_drop (float): The drop-out rate after the linear projection.
            Default: 0.0.

    """

    def __init__(self,
                 embed_dims,
                 num_heads=8,
                 qkv_bias=False,
                 attn_drop=0.,
                 proj_drop=0.):
        super(MultiheadAttention, self).__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dims // num_heads
        self.embed_dims = embed_dims
        assert self.head_dim * num_heads == embed_dims, \
            'embed_dim must be divisible by num_heads'

        self.scale = self.head_dim**-0.5

        self.qkv = nn.Linear(embed_dims, embed_dims * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(embed_dims, embed_dims)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads,
                                  C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[
            2]  # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class TransformerEncoderLayer(nn.Module):
    """Implements one encoder layer in Vision Transformer.

    Args:
        embed_dims (int): The feature dimension. Same as `FFN`.
        num_heads (int): Parallel attention heads.
        feedforward_channels (int): The hidden dimension for FFNs.
        qkv_bias (bool): Whether to add bias in qkv. Default False.
        attn_drop (float): The drop out rate after attention layer.
            Default 0.0.
        proj_drop (float): Probability of an element to be zeroed
            after the feed forward layer. Default 0.0.
        act_cfg (dict): The activation config for FFNs. Defalut GELU.
        norm_cfg (dict): Config dict for normalization layer. Default
            layer normalization.
        num_fcs (int): The number of fully-connected layers for FFNs.
            Default 2.
    """

    def __init__(self,
                 embed_dims,
                 num_heads,
                 feedforward_channels,
                 qkv_bias=False,
                 attn_drop=0.,
                 proj_drop=0.,
                 act_cfg=dict(type='GELU'),
                 norm_cfg=dict(type='LN'),
                 num_fcs=2):
        super(TransformerEncoderLayer, self).__init__()
        self.norm1_name, norm1 = build_norm_layer(
            norm_cfg, embed_dims, postfix=1)
        self.add_module(self.norm1_name, norm1)
        self.attn = MultiheadAttention(
            embed_dims,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=proj_drop)

        self.norm2_name, norm2 = build_norm_layer(
            norm_cfg, embed_dims, postfix=2)
        self.add_module(self.norm2_name, norm2)
        self.mlp = FFN(embed_dims, feedforward_channels, num_fcs, act_cfg,
                       proj_drop)

    @property
    def norm1(self):
        return getattr(self, self.norm1_name)

    @property
    def norm2(self):
        return getattr(self, self.norm2_name)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding.

    Args:
        img_size (int | tuple): The size of input image.
        patch_size (int): The size of one patch
        in_channels (int): The num of input channels.
        embed_dim (int): The dimensions of embedding.
        conv_cfg (dict | None): The config dict for conv layers.
            Default: None.
    """

    def __init__(self,
                 img_size=224,
                 patch_size=16,
                 in_channels=3,
                 embed_dim=768,
                 conv_cfg=None):
        super(PatchEmbed, self).__init__()
        if isinstance(img_size, int):
            img_size = tuple(repeat(img_size, 2))
        elif isinstance(img_size, tuple):
            if len(img_size) == 1:
                img_size = tuple(repeat(img_size[0], 2))
            assert len(img_size) == 2, \
                f'The size of image should have length 1 or 2, ' \
                f'but got {len(img_size)}'

        self.img_size = img_size
        self.patch_size = tuple(repeat(patch_size, 2))

        num_patches = (self.img_size[1] // self.patch_size[1]) * (
            self.img_size[0] // self.patch_size[0])
        assert num_patches * self.patch_size[0] * self.patch_size[1] == \
               self.img_size[0] * self.img_size[1], \
               'The image size H*W must be divisible by patch size'
        self.num_patches = num_patches

        self.projection = build_conv_layer(
            conv_cfg,
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't " \
            f'match model ({self.img_size[0]}*{self.img_size[1]}).'
        # The output size is (B, N, D), where N=H*W/P/P, D is embid_dim
        x = self.projection(x).flatten(2).transpose(1, 2)
        return x


class HybridEmbed(PatchEmbed):
    """ CNN Feature Map Embedding.
        Extract feature map from CNN, flatten, project to embedding dim.
    """

    def __init__(self,
                 backbone,
                 img_size=224,
                 in_channels=3,
                 embed_dim=768,
                 conv_cfg=None):
        assert isinstance(backbone, nn.Module)
        self.backbone = backbone
        if isinstance(img_size, int):
            img_size = tuple(repeat(img_size, 2))
        elif isinstance(img_size, tuple):
            if len(img_size) == 1:
                img_size = tuple(repeat(img_size[0], 2))
            assert len(img_size) == 2, \
                f'The size of image should have length 1 or 2, ' \
                f'but got {len(img_size)}'

        # Get the output dim of the backbone
        with torch.no_grad():
            # FIXME this is hacky, but most reliable way of
            # determining the exact dim of the output feature
            # map for all networks, the feature metadata has
            # reliable channel and stride info, but using
            # stride to calc feature dim requires info about
            # padding of each stage that isn't captured.
            training = backbone.training
            if training:
                backbone.eval()
            o = self.backbone(
                torch.zeros(1, in_channels, img_size[0], img_size[1]))
            if isinstance(o, (list, tuple)):
                # last feature if backbone outputs list/tuple of features
                o = o[-1]
            feature_size = o.shape[-2:]
            feature_dim = o.shape[1]
            backbone.train(training)

        super(HybridEmbed, self).__init__(
            img_size=feature_size,
            patch_size=1,
            in_channels=feature_dim,
            embed_dim=embed_dim,
            conv_cfg=conv_cfg)

    def forward(self, x):
        x = self.backbone(x)
        if isinstance(x, (list, tuple)):
            x = x[
                -1]  # last feature if backbone outputs list/tuple of features
        x = self.projection(x).flatten(2).transpose(1, 2)
        return x


@BACKBONES.register_module()
class VisionTransformer(BaseBackbone):
    """ Vision Transformer

    A PyTorch impl of : `An Image is Worth 16x16 Words:
    Transformers for Image Recognition at Scale`  -
        https://arxiv.org/abs/2010.11929

    Args:
        img_size (int, tuple): input image size
        patch_size (int, tuple): patch size
        in_channels (int): number of input channels
        num_classes (int): number of classes for classification head
        embed_dim (int): embedding dimension
        num_layers (int): depth of transformer
        num_heads (int): number of attention heads
        mlp_ratio (int): ratio of mlp hidden dim to embedding dim
        qkv_bias (bool): enable bias for qkv if True
        qk_scale (float): override default qk scale of head_dim ** -0.5 if set
                representation_size (Optional[int]): enable and set
                representation
                layer (pre-logits) to this value if set
                drop_rate (float): dropout rate
                attn_drop_rate (float): attention dropout rate
                drop_path_rate (float): stochastic depth rate
                hybrid_backbone (nn.Module): CNN backbone to use in-place of
                PatchEmbed module
                norm_layer: (nn.Module): normalization layer
    """

    def __init__(self,
                 num_layers,
                 embed_dim,
                 num_heads,
                 img_size,
                 patch_size,
                 in_channels,
                 feedforward_channels,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 qkv_bias=True,
                 hybrid_backbone=None,
                 norm_cfg=dict(type='LN'),
                 act_cfg=dict(type='GELU'),
                 num_fcs=2):
        super(VisionTransformer, self).__init__()
        self.embed_dim = embed_dim

        if hybrid_backbone is not None:
            self.patch_embed = HybridEmbed(
                hybrid_backbone,
                img_size=img_size,
                in_channels=in_channels,
                embed_dim=embed_dim)
        else:
            self.patch_embed = PatchEmbed(
                img_size=img_size,
                patch_size=patch_size,
                in_channels=in_channels,
                embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, embed_dim))
        self.drop_after_pos = nn.Dropout(p=drop_rate)

        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(
                TransformerEncoderLayer(
                    embed_dim,
                    num_heads,
                    feedforward_channels,
                    qkv_bias=qkv_bias,
                    attn_drop=attn_drop_rate,
                    proj_drop=drop_rate,
                    act_cfg=act_cfg,
                    norm_cfg=norm_cfg,
                    num_fcs=num_fcs))

        self.norm1_name, norm1 = build_norm_layer(
            norm_cfg, embed_dim, postfix=1)
        self.add_module(self.norm1_name, norm1)

        # trunc_normal_(self.pos_embed, std=.02)
        # trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @property
    def norm1(self):
        return getattr(self, self.norm1_name)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(
            B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.drop_after_pos(x)

        for layer in self.layers:
            x = layer(x)

        x = self.norm1(x)[:, 0]
        return x
