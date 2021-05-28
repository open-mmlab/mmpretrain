import torch
import torch.nn as nn
from mmcv.cnn import build_norm_layer
from mmcv.cnn.bricks.conv import build_conv_layer
from mmcv.cnn.bricks.registry import ATTENTION
from mmcv.cnn.bricks.transformer import FFN, build_dropout
from mmcv.runner.base_module import BaseModule, ModuleList

from ..builder import BACKBONES
from ..utils import to_2tuple
from .base_backbone import BaseBackbone


class PatchMerging(BaseModule):

    def __init__(self, input_resolution, embed_dims, norm_cfg=dict(type='LN')):
        super().__init__()
        self.input_resolution = input_resolution
        self.embed_dims = embed_dims
        self.reduction = nn.Linear(4 * embed_dims, 2 * embed_dims, bias=False)
        self.sampler = nn.Unfold(kernel_size=(2, 2), stride=2)
        if norm_cfg is not None:
            self.norm = build_norm_layer(norm_cfg, 4 * embed_dims)[1]
        else:
            self.norm = None

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, 'input feature has wrong size'
        assert H % 2 == 0 and W % 2 == 0, f'x size ({H}*{W}) are not even.'

        x = x.view(B, H, W, C).permute([0, 3, 1, 2])  # B, C, H, W

        # Use nn.Unfold to merge patch. About 25% faster than original method,
        # but need to modify pretrained model for compatibility
        x = self.sampler(x)  # B, 4*C, H/2*W/2
        x = x.transpose(1, 2)  # B, H/2*W/2, 4*C

        x = self.norm(x) if self.norm else x
        x = self.reduction(x)

        return x


class PatchEmbed(BaseModule):
    """Image to Patch Embedding.

    Args:
        img_size (int | tuple): The size of input image.
        patch_size (int): The size of one patch
        in_channels (int): The num of input channels.
        embed_dims (int): The dimensions of embedding.
        norm_cfg (dict, optional): Config dict for normalization layer.
        conv_cfg (dict, optional): The config dict for conv layers.
            Default: None.
        init_cfg (`mmcv.dict`, optional): The Config for initialization.
            Default: None.
    """

    def __init__(self,
                 img_size=224,
                 patch_size=16,
                 in_channels=3,
                 embed_dims=768,
                 norm_cfg=dict(type='LN'),
                 conv_cfg=None,
                 init_cfg=None):
        super(PatchEmbed, self).__init__(init_cfg)
        if isinstance(img_size, int):
            img_size = to_2tuple(img_size)
        elif isinstance(img_size, tuple):
            if len(img_size) == 1:
                img_size = to_2tuple(img_size[0])
            assert len(img_size) == 2, \
                f'The size of image should have length 1 or 2, ' \
                f'but got {len(img_size)}'

        self.img_size = img_size
        self.patch_size = to_2tuple(patch_size)

        patches_resolution = [
            img_size[0] // self.patch_size[0],
            img_size[1] // self.patch_size[1]
        ]
        num_patches = patches_resolution[0] * patches_resolution[1]
        assert num_patches * self.patch_size[0] * self.patch_size[1] == \
               self.img_size[0] * self.img_size[1], \
               'The image size H*W must be divisible by patch size'
        self.patches_resolution = patches_resolution
        self.num_patches = num_patches

        # Use conv layer to embed
        self.projection = build_conv_layer(
            conv_cfg,
            in_channels,
            embed_dims,
            kernel_size=patch_size,
            stride=patch_size)

        if norm_cfg is not None:
            self.norm = build_norm_layer(norm_cfg, embed_dims)[1]
        else:
            self.norm = None

    def forward(self, x):
        _, _, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't " \
            f'match model ({self.img_size[0]}*{self.img_size[1]}).'
        # The output size is (B, N, D), where N=H*W/P/P, D is embid_dim
        x = self.projection(x).flatten(2).transpose(1, 2)

        if self.norm is not None:
            x = self.norm(x)

        return x


@ATTENTION.register_module()
class WindowMSA(BaseModule):
    """Window based multi-head self attention (W-MSA) module with relative
    position bias.

    Args:
        embed_dims (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to q, k, v.
            Default: True
        qk_scale (float | None, optional): Override default qk scale of
            head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight.
            Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self,
                 embed_dims,
                 window_size,
                 num_heads,
                 qkv_bias=True,
                 qk_scale=None,
                 attn_drop=0.,
                 proj_drop=0.):

        super().__init__()
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
        rel_index_coords_h = self.double_step_seq(2 * Wh - 1, Wh, 1, Wh)
        rel_index_coords_w = self.double_step_seq(2 * Ww - 1, Ww, 1, Ww)
        rel_position_index = rel_index_coords_h + rel_index_coords_w.T
        rel_position_index = rel_position_index.flip(1).contiguous()
        self.register_buffer('relative_position_index', rel_position_index)

        self.qkv = nn.Linear(embed_dims, embed_dims * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(embed_dims, embed_dims)
        self.proj_drop = nn.Dropout(proj_drop)

        self.softmax = nn.Softmax(dim=-1)

    def init_weights(self):
        super(WindowMSA, self).init_weights()

        # FIXME: trunc_normal_ is added after pt1.8, use previous version
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

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

    def __init__(self,
                 embed_dims,
                 num_heads,
                 input_resolution,
                 window_size,
                 shift_size=0,
                 qkv_bias=True,
                 qk_scale=None,
                 attn_drop=0,
                 proj_drop=0,
                 dropout_layer=dict(type='DropPath', drop_prob=0.)):
        super().__init__()

        self.w_msa = WindowMSA(embed_dims, to_2tuple(window_size), num_heads,
                               qkv_bias, qk_scale, attn_drop, proj_drop)

        self.embed_dims = embed_dims
        self.input_resolution = input_resolution
        self.shift_size = shift_size
        self.window_size = window_size
        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, don't partition
            self.shift_size = 0
            self.window_size = min(self.input_resolution)

        self.drop = build_dropout(dropout_layer)

        if self.shift_size > 0:
            # calculate attention mask for SW-MSA
            H, W = self.input_resolution
            img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
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
            # TODO: what -100 means?
        else:
            attn_mask = None

        self.register_buffer('attn_mask', attn_mask)

    def forward(self, query, key=None, value=None, residual=None, **kwargs):
        if key is None:
            key = query
        if value is None:
            value = key
        if residual is None:
            residual = query

        H, W = self.input_resolution
        B, L, C = query.shape
        assert L == H * W, 'input feature has wrong size'
        query = query.view(B, H, W, C)

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
        shifted_x = self.window_reverse(attn_windows)
        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(
                shifted_x,
                shifts=(self.shift_size, self.shift_size),
                dims=(1, 2))
        else:
            x = shifted_x
        x = x.view(B, H * W, C)

        x = residual + self.drop(x)
        return x

    def window_reverse(self, windows):
        H, W = self.input_resolution
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


class SwinBlock(BaseModule):

    def __init__(self,
                 embed_dims,
                 input_resolution,
                 num_heads,
                 window_size=7,
                 shift=False,
                 mlp_ratio=4.,
                 drop_path=0.,
                 attn_cfgs=dict(),
                 ffn_cfgs=dict(),
                 act_cfg=dict(type='GELU'),
                 norm_cfg=dict(type='LN'),
                 init_cfg=None):

        super(SwinBlock, self).__init__(init_cfg)

        _attn_cfgs = {
            'embed_dims': embed_dims,
            'num_heads': num_heads,
            'input_resolution': input_resolution,
            'shift_size': window_size // 2 if shift else 0,
            'window_size': window_size,
            **attn_cfgs
        }
        self.norm1 = build_norm_layer(norm_cfg, embed_dims)[1]
        self.attn = ShiftWindowMSA(**_attn_cfgs)

        _ffn_cfgs = {
            'embed_dims': embed_dims,
            'feedforward_channels': int(embed_dims * mlp_ratio),
            'num_fcs': 2,
            'ffn_drop': 0,
            'dropout_layer': dict(type='DropPath', drop_prob=drop_path),
            'act_cfg': act_cfg,
            **ffn_cfgs
        }
        self.norm2 = build_norm_layer(norm_cfg, embed_dims)[1]
        self.ffn = FFN(**_ffn_cfgs)

    def forward(self, x):
        residual = x
        x = self.norm1(x)
        x = self.attn(x, residual=residual)

        residual = x
        x = self.norm2(x)
        x = self.ffn(x, residual=residual)
        return x


class SwinBlockSequence(BaseModule):

    def __init__(self,
                 embed_dims,
                 input_resolution,
                 depth,
                 num_heads,
                 downsample=None,
                 drop_path=0.,
                 norm_cfg=dict(type='LN'),
                 block_cfg=dict(),
                 init_cfg=None):
        super().__init__(init_cfg)

        if not isinstance(drop_path, list):
            drop_path = [drop_path] * depth

        if not isinstance(block_cfg, list):
            block_cfg = [block_cfg] * depth

        self.blocks = ModuleList()
        for i in range(depth):
            _block_cfg = {
                'embed_dims': embed_dims,
                'input_resolution': input_resolution,
                'num_heads': num_heads,
                'shift': False if i % 2 == 0 else True,
                'drop_path': drop_path[i],
                **block_cfg[i]
            }
            block = SwinBlock(**_block_cfg)
            self.blocks.append(block)

        if downsample:
            self.downsample = PatchMerging(
                input_resolution, embed_dims=embed_dims, norm_cfg=norm_cfg)
        else:
            self.downsample = None

    def forward(self, query):
        for block in self.blocks:
            query = block(query)

        if self.downsample:
            query = self.downsample(query)
        return query


@BACKBONES.register_module()
class SwinTransformer(BaseBackbone):
    """ Swin Transformer
    A PyTorch impl of : `Swin Transformer:
    Hierarchical Vision Transformer using Shifted Windows`  -
        https://arxiv.org/abs/2103.14030

    Inspiration from
    https://github.com/microsoft/Swin-Transformer

    Args:
        arch (str | dict): Swin Transformer architecture
            Default: 'T'
        img_size (int | tuple): The size of input image.
            Default: 224
        in_channels (int): The num of input channels.
            Default: 3
        drop_rate (float): Dropout rate.
            Default: 0
        drop_path_rate (float): Stochastic depth rate.
            Default: 0.1
        ape (bool): If True, add absolute position embedding to the patch
            embedding. Default: False
        norm_cfg(dict, optional): Config dict for normalization layer at end of
            backone. Default: dict(type='LN')
        stage_cfg(dict, optional): Extra config dict for stages.
        patch_cfg(dict, optional): Extra config dict for patch embedding.
        init_cfg(dict, optional): Extra config dict for model initialization.
    """
    arch_zoo = {
        #     depth num_heads downsample
        **dict.fromkeys(['t', 'tiny'],
                        {'embed_dims': 96,
                         'depths':     [2, 2,  6,  2],
                         'num_heads':  [3, 6, 12, 24]}),
        **dict.fromkeys(['s', 'small'],
                        {'embed_dims': 96,
                         'depths':     [2, 2, 18,  2],
                         'num_heads':  [3, 6, 12, 24]}),
        **dict.fromkeys(['b', 'base'],
                        {'embed_dims': 128,
                         'depths':     [2, 2, 18,  2],
                         'num_heads':  [4, 8, 16, 32]}),
        **dict.fromkeys(['l', 'large'],
                        {'embed_dims': 192,
                         'depths':     [2,  2, 18,  2],
                         'num_heads':  [6, 12, 24, 48]}),
    }  # yapf: disable

    def __init__(self,
                 arch='T',
                 img_size=224,
                 in_channels=3,
                 drop_rate=0.,
                 drop_path_rate=0.1,
                 ape=False,
                 norm_cfg=dict(type='LN'),
                 stage_cfg=dict(),
                 patch_cfg=dict(),
                 init_cfg=None):
        super(SwinTransformer, self).__init__(init_cfg)

        arch = arch.lower()
        if isinstance(arch, str):
            assert arch in set(self.arch_zoo), \
                f'Arch {arch} is not in default archs {set(self.arch_zoo)}'
            self.arch_settings = self.arch_zoo[arch]
        else:
            essential_keys = {'embed_dims', 'depths', 'num_head'}
            assert isinstance(arch, dict) and set(arch) == essential_keys, \
                f'Custom arch needs a dict with keys {essential_keys}'
            self.arch_settings = arch

        self.embed_dims = self.arch_settings['embed_dims']
        self.depths = self.arch_settings['depths']
        self.num_heads = self.arch_settings['num_heads']
        self.num_layers = len(self.depths)
        self.ape = ape
        self.num_features = int(self.embed_dims * 2**(self.num_layers - 1))

        _patch_cfg = dict(
            img_size=img_size,
            in_channels=in_channels,
            embed_dims=self.embed_dims,
            patch_size=4,
            **patch_cfg)
        self.patch_embed = PatchEmbed(**_patch_cfg)
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        if self.ape:
            self.absolute_pos_embed = nn.Parameter(
                torch.zeros(1, num_patches, self.embed_dims))

        self.drop_after_pos = nn.Dropout(p=drop_rate)

        # stochastic depth
        total_depth = sum(self.depths)
        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, total_depth)
        ]  # stochastic depth decay rule

        self.stages = ModuleList()
        scale_factor = 1
        for i, (depth,
                num_heads) in enumerate(zip(self.depths, self.num_heads)):
            downsample = True if i < self.num_layers - 1 else False
            input_resolution = [i // scale_factor for i in patches_resolution]
            _stage_cfg = {
                'embed_dims': self.embed_dims * scale_factor,
                'depth': depth,
                'num_heads': num_heads,
                'downsample': downsample,
                'input_resolution': input_resolution,
                'drop_path': dpr[:depth],
                **stage_cfg
            }

            stage = SwinBlockSequence(**_stage_cfg)
            self.stages.append(stage)

            dpr = dpr[depth:]
            if downsample:
                scale_factor *= 2

        if norm_cfg is not None:
            self.norm = build_norm_layer(norm_cfg, self.num_features)[1]
        else:
            self.norm = None

    def init_weights(self, pretrained=None):
        super().init_weights(pretrained)

        if pretrained is None and self.ape:
            # FIXME: trunc_normal_ is added after pt1.8, use previous version
            nn.init.trunc_normal_(self.absolute_pos_embed, std=0.02)
        # FIXME: temporary init method, replace it with init_cfg
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.drop_after_pos(x)

        for stage in self.stages:
            x = stage(x)

        x = self.norm(x) if self.norm else x

        return x.transpose(1, 2)
