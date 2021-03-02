import torch.nn as nn
from mmcv.cnn import build_norm_layer

# from ..builder import BACKBONES
from ..utils import FFN

# from .base_backbone import BaseBackbone

# import torch.utils.checkpoint as cp
# from mmcv.cnn import (ConvModule, build_conv_layer, build_norm_layer,
#   constant_init, kaiming_init)
# from mmcv.utils.parrots_wrapper import _BatchNorm


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
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dims // num_heads
        assert self.head_dim * num_heads == self.embed_dim, \
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


# TODO: 没搞完
# class PatchEmbed(nn.Module):
#     """ Image to Patch Embedding
#     """

#     def __init__(self, img_size=224, patch_size=16,
# in_chans=3, embed_dim=768):
#         super().__init__()
#         img_size = to_2tuple(img_size)
#         patch_size = to_2tuple(patch_size)
#         num_patches = (img_size[1] // patch_size[1]) * (
#             img_size[0] // patch_size[0])
#         self.img_size = img_size
#         self.patch_size = patch_size
#         self.num_patches = num_patches

#         self.proj = nn.Conv2d(
#             in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

#     def forward(self, x):
#         B, C, H, W = x.shape
#         # FIXME look at relaxing size constraints
#         assert H == self.img_size[0] and W == self.img_size[1], \
#             f"Input image size ({H}*{W}) doesn't match model
#  ({self.img_size[0]}*{self.img_size[1]})."
#         x = self.proj(x).flatten(2).transpose(1, 2)
#         return x
