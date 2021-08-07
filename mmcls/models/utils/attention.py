import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn.bricks.transformer import build_dropout
from mmcv.cnn.utils.weight_init import trunc_normal_
from mmcv.runner.base_module import BaseModule

from ..builder import ATTENTION
from .helpers import to_2tuple


class WindowMSA(BaseModule):
    """Window based multi-head self-attention (W-MSA) module with relative
    position bias.

    Args:
        embed_dims (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional): If True, add a learnable bias to q, k, v.
            Defaults to True.
        qk_scale (float | None, optional): Override default qk scale of
            head_dim ** -0.5 if set. Defaults to None.
        attn_drop (float, optional): Dropout ratio of attention weight.
            Defaults to 0.
        proj_drop (float, optional): Dropout ratio of output. Defaults to 0.
        init_cfg (dict, optional): The extra config for initialization.
            Defaults to None.
    """

    def __init__(self,
                 embed_dims,
                 window_size,
                 num_heads,
                 qkv_bias=True,
                 qk_scale=None,
                 attn_drop=0.,
                 proj_drop=0.,
                 init_cfg=None):

        super().__init__(init_cfg)
        self.embed_dims = embed_dims
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_embed_dims = embed_dims // num_heads
        self.scale = qk_scale or head_embed_dims**-0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1),
                        num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # About 2x faster than original impl
        Wh, Ww = self.window_size
        rel_index_coords = self.double_step_seq(2 * Ww - 1, Wh, 1, Ww)
        rel_position_index = rel_index_coords + rel_index_coords.T
        rel_position_index = rel_position_index.flip(1).contiguous()
        self.register_buffer('relative_position_index', rel_position_index)

        self.qkv = nn.Linear(embed_dims, embed_dims * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(embed_dims, embed_dims)
        self.proj_drop = nn.Dropout(proj_drop)

        self.softmax = nn.Softmax(dim=-1)

    def init_weights(self):
        super(WindowMSA, self).init_weights()

        trunc_normal_(self.relative_position_bias_table, std=0.02)

    def forward(self, x, mask=None):
        """
        Args:

            x (tensor): input features with shape of (num_windows*B, N, C)
            mask (tensor, Optional): mask with shape of (num_windows, Wh*Ww,
                Wh*Ww), value should be between (-inf, 0].
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads,
                                  C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[
            2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)].view(
                self.window_size[0] * self.window_size[1],
                self.window_size[0] * self.window_size[1],
                -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(
            2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N,
                             N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    @staticmethod
    def double_step_seq(step1, len1, step2, len2):
        seq1 = torch.arange(0, step1 * len1, step1)
        seq2 = torch.arange(0, step2 * len2, step2)
        return (seq1[:, None] + seq2[None, :]).reshape(1, -1)


@ATTENTION.register_module()
class ShiftWindowMSA(BaseModule):
    """Shift Window Multihead Self-Attention Module.

    Args:
        embed_dims (int): Number of input channels.
        input_resolution (Tuple[int, int]): The resolution of the input feature
            map.
        num_heads (int): Number of attention heads.
        window_size (int): The height and width of the window.
        shift_size (int, optional): The shift step of each window towards
            right-bottom. If zero, act as regular window-msa. Defaults to 0.
        qkv_bias (bool, optional): If True, add a learnable bias to q, k, v.
            Default: True
        qk_scale (float | None, optional): Override default qk scale of
            head_dim ** -0.5 if set. Defaults to None.
        attn_drop (float, optional): Dropout ratio of attention weight.
            Defaults to 0.0.
        proj_drop (float, optional): Dropout ratio of output. Defaults to 0.
        dropout_layer (dict, optional): The dropout_layer used before output.
            Defaults to dict(type='DropPath', drop_prob=0.).
        auto_pad (bool, optional): Auto pad the feature map to be divisible by
            window_size, Defaults to False.
        init_cfg (dict, optional): The extra config for initialization.
            Default: None.
    """

    def __init__(self,
                 embed_dims,
                 input_resolution,
                 num_heads,
                 window_size,
                 shift_size=0,
                 qkv_bias=True,
                 qk_scale=None,
                 attn_drop=0,
                 proj_drop=0,
                 dropout_layer=dict(type='DropPath', drop_prob=0.),
                 auto_pad=False,
                 init_cfg=None):
        super().__init__(init_cfg)

        self.embed_dims = embed_dims
        self.input_resolution = input_resolution
        self.shift_size = shift_size
        self.window_size = window_size
        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, don't partition
            self.shift_size = 0
            self.window_size = min(self.input_resolution)

        self.w_msa = WindowMSA(embed_dims, to_2tuple(self.window_size),
                               num_heads, qkv_bias, qk_scale, attn_drop,
                               proj_drop)

        self.drop = build_dropout(dropout_layer)

        H, W = self.input_resolution
        # Handle auto padding
        self.auto_pad = auto_pad
        if self.auto_pad:
            self.pad_r = (self.window_size -
                          W % self.window_size) % self.window_size
            self.pad_b = (self.window_size -
                          H % self.window_size) % self.window_size
            self.H_pad = H + self.pad_b
            self.W_pad = W + self.pad_r
        else:
            H_pad, W_pad = self.input_resolution
            assert H_pad % self.window_size + W_pad % self.window_size == 0,\
                f'input_resolution({self.input_resolution}) is not divisible '\
                f'by window_size({self.window_size}). Please check feature '\
                f'map shape or set `auto_pad=True`.'
            self.H_pad, self.W_pad = H_pad, W_pad
            self.pad_r, self.pad_b = 0, 0

        if self.shift_size > 0:
            # calculate attention mask for SW-MSA
            img_mask = torch.zeros((1, self.H_pad, self.W_pad, 1))  # 1 H W 1
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size,
                              -self.shift_size), slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size,
                              -self.shift_size), slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            # nW, window_size, window_size, 1
            mask_windows = self.window_partition(img_mask)
            mask_windows = mask_windows.view(
                -1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0,
                                              float(-100.0)).masked_fill(
                                                  attn_mask == 0, float(0.0))
        else:
            attn_mask = None

        self.register_buffer('attn_mask', attn_mask)

    def forward(self, query):
        H, W = self.input_resolution
        B, L, C = query.shape
        assert L == H * W, 'input feature has wrong size'
        query = query.view(B, H, W, C)

        if self.pad_r or self.pad_b:
            query = F.pad(query, (0, 0, 0, self.pad_r, 0, self.pad_b))

        # cyclic shift
        if self.shift_size > 0:
            shifted_query = torch.roll(
                query,
                shifts=(-self.shift_size, -self.shift_size),
                dims=(1, 2))
        else:
            shifted_query = query

        # nW*B, window_size, window_size, C
        query_windows = self.window_partition(shifted_query)
        # nW*B, window_size*window_size, C
        query_windows = query_windows.view(-1, self.window_size**2, C)

        # W-MSA/SW-MSA (nW*B, window_size*window_size, C)
        attn_windows = self.w_msa(query_windows, mask=self.attn_mask)

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size,
                                         self.window_size, C)

        # B H' W' C
        shifted_x = self.window_reverse(attn_windows, self.H_pad, self.W_pad)
        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(
                shifted_x,
                shifts=(self.shift_size, self.shift_size),
                dims=(1, 2))
        else:
            x = shifted_x

        if self.pad_r or self.pad_b:
            x = x[:, :H, :W, :].contiguous()

        x = x.view(B, H * W, C)

        x = self.drop(x)
        return x

    def window_reverse(self, windows, H, W):
        window_size = self.window_size
        B = int(windows.shape[0] / (H * W / window_size / window_size))
        x = windows.view(B, H // window_size, W // window_size, window_size,
                         window_size, -1)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
        return x

    def window_partition(self, x):
        B, H, W, C = x.shape
        window_size = self.window_size
        x = x.view(B, H // window_size, window_size, W // window_size,
                   window_size, C)
        windows = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        windows = windows.view(-1, window_size, window_size, C)
        return windows
