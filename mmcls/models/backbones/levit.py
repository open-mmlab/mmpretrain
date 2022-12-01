import itertools

import torch.nn as nn
import torch

import mmcv
from mmcv.cnn import build_conv_layer, build_norm_layer, Linear
from mmengine.model import BaseModule

from mmcls.models.backbones.base_backbone import BaseBackbone
from mmcls.models.heads.levit_head import LeViTClsHead
from mmcls.registry import MODELS


# from mmcls.models.heads import

class hybrid_cnn(BaseModule):
    def __init__(self,
                 embed_dim,
                 kernel_size=3,
                 stride=2,
                 pad=1,
                 dilation=1,
                 groups=1,
                 bn_weight_init=1,
                 activation=nn.Hardswish,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 init_cfg=None,
                 ):
        super(hybrid_cnn, self).__init__(init_cfg=init_cfg)

        self.input_channels = [3, embed_dim // 8, embed_dim // 4, embed_dim // 2]
        self.output_channels = [embed_dim // 8, embed_dim // 4, embed_dim // 2, embed_dim]
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg

        self.patch_embed = nn.Sequential()

        # self.patch_embed.add_module()
        for i in range(len(self.input_channels)):
            conv_bn = build_conv_bn(self.input_channels[i],
                                    self.output_channels[i],
                                    kernel_size=kernel_size,
                                    stride=stride,
                                    pad=pad,
                                    dilation=dilation,
                                    groups=groups,
                                    bn_weight_init=bn_weight_init,
                                    conv_cfg=conv_cfg,
                                    norm_cfg=norm_cfg,
                                    )
            self.patch_embed.add_module('%d' % (2 * i), conv_bn)
            if i < len(self.input_channels) - 1:
                self.patch_embed.add_module('%d' % (i * 2 + 1), activation())

    def forward(self, x):
        x = self.patch_embed(x)
        return x


def build_conv_bn(
        in_channel,
        out_channel,
        kernel_size=3,
        stride=2,
        pad=1,
        dilation=1,
        groups=1,
        bn_weight_init=1,
        conv_cfg=None,
        norm_cfg=dict(type='BN'),
):
    # super(Conv_BN, self).__init__()
    conv_cfg = conv_cfg
    norm_cfg = norm_cfg
    input_channel = in_channel
    output_channel = out_channel
    _, bn = build_norm_layer(norm_cfg, output_channel)
    torch.nn.init.constant_(bn.weight, bn_weight_init)
    torch.nn.init.constant_(bn.bias, 0)
    conv = build_conv_layer(
        conv_cfg,
        input_channel,
        output_channel,
        kernel_size=kernel_size,
        stride=stride,
        padding=pad,
        dilation=dilation,
        groups=groups,
        bias=False
    )
    w = bn.weight / (bn.running_var + bn.eps) ** 0.5
    w = conv.weight * w[:, None, None, None]
    b = bn.bias - bn.running_mean * bn.weight / \
        (bn.running_var + bn.eps) ** 0.5
    m = build_conv_layer(conv_cfg, w.size(1) * conv.groups, w.size(
        0), w.shape[2:], stride=conv.stride, padding=conv.padding, dilation=conv.dilation,
                         groups=conv.groups)
    m.weight.data.copy_(w)
    m.bias.data.copy_(b)
    return m


def build_linear_bn(
        in_feature,
        out_feature,
        bn_weight_init=1,
        norm_cfg=dict(type='BN1d')
):
    linear = Linear(in_feature, out_feature, bias=False)
    _, bn = build_norm_layer(norm_cfg, out_feature)
    torch.nn.init.constant_(bn.weight, bn_weight_init)
    torch.nn.init.constant_(bn.bias, 0)
    w = bn.weight / (bn.running_var + bn.eps) ** 0.5
    w = linear.weight * w[:, None]
    b = bn.bias - bn.running_mean * bn.weight / \
        (bn.running_var + bn.eps) ** 0.5
    m = Linear(w.size(1), w.size(0))
    m.weight.data.copy_(w)
    m.bias.data.copy_(b)
    return m


class Residual(BaseModule):
    def __init__(self, block, drop):
        super(Residual, self).__init__()
        self.block = block
        self.drop = drop

    def forward(self, x):
        if self.training and self.drop > 0:
            return x + self.block(x) * torch.rand(x.size(0), 1, 1,
                                                  device=x.device).ge_(self.drop).div(1 - self.drop).detach()
        else:
            return x + self.block(x)  # add操作


class Attention(BaseModule):
    def __init__(self,
                 dim,
                 key_dim,
                 num_heads=8,
                 attn_ratio=4,
                 activation=None,
                 resolution=14,
                 ):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.scale = key_dim ** -0.5
        self.key_dim = key_dim
        self.nh_kd = nh_kd = key_dim * num_heads
        self.d = int(attn_ratio * key_dim)
        self.dh = int(attn_ratio * key_dim) * num_heads
        self.attn_ratio = attn_ratio
        h = self.dh + nh_kd * 2
        self.qkv = build_linear_bn(dim, h)
        self.proj = nn.Sequential(activation(), build_linear_bn(
            self.dh, dim, bn_weight_init=0))

        points = list(itertools.product(range(resolution), range(resolution)))
        N = len(points)
        attention_offsets = {}
        idxs = []
        for p1 in points:
            for p2 in points:
                offset = (abs(p1[0] - p2[0]), abs(p1[1] - p2[1]))
                if offset not in attention_offsets:
                    attention_offsets[offset] = len(attention_offsets)
                idxs.append(attention_offsets[offset])
        self.attention_biases = torch.nn.Parameter(
            torch.zeros(num_heads, len(attention_offsets)))
        self.register_buffer('attention_bias_idxs',
                             torch.LongTensor(idxs).view(N, N))

    @torch.no_grad()
    def train(self, mode=True):
        super(Attention, self).train(mode)
        if mode and hasattr(self, 'ab'):
            del self.ab
        else:
            self.ab = self.attention_biases[:, self.attention_bias_idxs]

    def forward(self, x):  # x (B,N,C)
        B, N, C = x.shape  # 2 196 128
        qkv = self.qkv(x)  # 2 196 128
        q, k, v = qkv.view(B, N, self.num_heads, -
        1).split([self.key_dim, self.key_dim, self.d], dim=3)  # q 2 196 4 16 ; k 2 196 4 16; v 2 196 4 32
        q = q.permute(0, 2, 1, 3)  # 2 4 196 16
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)

        attn = (
                (q @ k.transpose(-2, -1)) * self.scale  # 2 4 196 16 * 2 4 16 196 -> 2 4 196 196
                +
                (self.attention_biases[:, self.attention_bias_idxs]
                 if self.training else self.ab)
        )
        attn = attn.softmax(dim=-1)  # 2 4 196 196 -> 2 4 196 196
        x = (attn @ v).transpose(1, 2).reshape(B, N, self.dh)  # 2 4 196 196 * 2 4 196 32 -> 2 4 196 32 -> 2 196 128
        x = self.proj(x)
        return x


class MLP(nn.Sequential):
    def __init__(self,
                 embed_dim,
                 mlp_ratio,
                 mlp_activation,
                 ):
        super(MLP, self).__init__()
        h = embed_dim * mlp_ratio
        self.linear1 = build_linear_bn(embed_dim, h)
        self.activation = mlp_activation()
        self.linear2 = build_linear_bn(h, embed_dim, bn_weight_init=0)

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)
        return x


class Subsample(BaseModule):
    def __init__(self, stride, resolution):
        super(Subsample, self).__init__()
        self.stride = stride
        self.resolution = resolution

    def forward(self, x):
        B, N, C = x.shape  # 2 196 128
        x = x.view(B, self.resolution, self.resolution, C)[  # 2 196 128 -> 2 14 14 128  -> 2 7 7 128
            :, ::self.stride, ::self.stride].reshape(B, -1, C)  # # 2 14 14 128 > 2 49 128
        return x


class AttentionSubsample(nn.Sequential):
    def __init__(self, in_dim, out_dim, key_dim, num_heads=8,
                 attn_ratio=2,
                 activation=None,
                 stride=2,
                 resolution=14, resolution_=7):
        super(AttentionSubsample, self).__init__()
        self.num_heads = num_heads
        self.scale = key_dim ** -0.5
        self.key_dim = key_dim
        self.nh_kd = nh_kd = key_dim * num_heads
        self.d = int(attn_ratio * key_dim)
        self.dh = int(attn_ratio * key_dim) * self.num_heads
        self.attn_ratio = attn_ratio
        self.resolution_ = resolution_
        self.resolution_2 = resolution_ ** 2
        h = self.dh + nh_kd
        self.kv = build_linear_bn(in_dim, h)

        self.q = nn.Sequential(
            Subsample(stride, resolution),
            build_linear_bn(in_dim, nh_kd))
        self.proj = nn.Sequential(activation(), build_linear_bn(
            self.dh, out_dim))

        self.stride = stride
        self.resolution = resolution
        points = list(itertools.product(range(resolution), range(resolution)))
        points_ = list(itertools.product(
            range(resolution_), range(resolution_)))
        N = len(points)
        N_ = len(points_)
        attention_offsets = {}
        idxs = []
        for p1 in points_:
            for p2 in points:
                size = 1
                offset = (
                    abs(p1[0] * stride - p2[0] + (size - 1) / 2),
                    abs(p1[1] * stride - p2[1] + (size - 1) / 2))
                if offset not in attention_offsets:
                    attention_offsets[offset] = len(attention_offsets)
                idxs.append(attention_offsets[offset])
        self.attention_biases = torch.nn.Parameter(
            torch.zeros(num_heads, len(attention_offsets)))
        self.register_buffer('attention_bias_idxs',
                             torch.LongTensor(idxs).view(N_, N))

    @torch.no_grad()
    def train(self, mode=True):
        super(AttentionSubsample, self).train(mode)
        if mode and hasattr(self, 'ab'):
            del self.ab
        else:
            self.ab = self.attention_biases[:, self.attention_bias_idxs]

    def forward(self, x):
        B, N, C = x.shape
        k, v = self.kv(x).view(B, N, self.num_heads, -
        1).split([self.key_dim, self.d], dim=3)
        k = k.permute(0, 2, 1, 3)  # BHNC
        v = v.permute(0, 2, 1, 3)  # BHNC
        q = self.q(x).view(B, self.resolution_2, self.num_heads,
                           self.key_dim).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale + \
               (self.attention_biases[:, self.attention_bias_idxs]
                if self.training else self.ab)
        attn = attn.softmax(dim=-1)

        x = (attn @ v).transpose(1, 2).reshape(B, -1, self.dh)
        x = self.proj(x)
        return x


@MODELS.register_module()
class LeViT(BaseBackbone):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """

    def __init__(self,
                 img_size=224,
                 patch_size=16,
                 embed_dim=[192],
                 key_dim=[64],
                 depth=[12],
                 num_heads=[3],
                 attn_ratio=[2],
                 mlp_ratio=[2],
                 hybrid_backbone=None,
                 down_ops=[],
                 attention_activation=torch.nn.Hardswish,
                 mlp_activation=torch.nn.Hardswish,
                 drop_path=0,
                 out_indices=(2,)):
        super(LeViT, self).__init__()

        self.out_indices = out_indices

        self.num_features = embed_dim[-1]
        self.embed_dim = embed_dim
        if not hybrid_backbone:
            hybrid_backbone = hybrid_cnn(embed_dim[0], activation=torch.nn.Hardswish)
        self.patch_embed = hybrid_backbone
        self.blocks = [[]]
        self.size = []

        down_ops.append([''])
        resolution = img_size // patch_size
        for i, (ed, kd, dpth, nh, ar, mr, do) in enumerate(
                zip(embed_dim, key_dim, depth, num_heads, attn_ratio, mlp_ratio, down_ops)):
            for _ in range(dpth):
                self.blocks[-1].append(
                    Residual(Attention(
                        ed, kd, nh,
                        attn_ratio=ar,
                        activation=attention_activation,
                        resolution=resolution,
                    ), drop_path))
                if mr > 0:
                    # h = int(ed * mr)
                    self.blocks[-1].append(
                        Residual(
                            MLP(ed, mr, mlp_activation), drop_path))
            if do[0] == 'Subsample':
                # ('Subsample',key_dim, num_heads, attn_ratio, mlp_ratio, stride)
                self.size.append(resolution)
                resolution_ = (resolution - 1) // do[5] + 1
                self.blocks.append([])
                self.blocks[-1].append(
                    AttentionSubsample(
                        *embed_dim[i:i + 2], key_dim=do[1], num_heads=do[2],
                        attn_ratio=do[3],
                        activation=attention_activation,
                        stride=do[5],
                        resolution=resolution,
                        resolution_=resolution_))
                resolution = resolution_
                if do[4] > 0:  # mlp_ratio
                    self.blocks[-1].append(
                        Residual(nn.Sequential(
                            MLP(embed_dim[i + 1], do[4], mlp_activation)
                        ), drop_path))
        self.blocks = [nn.Sequential(*i) for i in self.blocks]
        self.size.append(resolution)
        self.stages = nn.Sequential()
        for i in range(len(self.blocks)):
            self.stages.add_module('%d' % i, self.blocks[i])

    @torch.jit.ignore
    def no_weight_decay(self):
        return {x for x in self.state_dict().keys() if 'attention_biases' in x}

    def forward(self, x):
        x = self.patch_embed(x)  # 2 3 224 224 -> 2 128 14 14
        x = x.flatten(2).transpose(1, 2)  # 2 128 14 14 -> 2 128 196 -> 2 196 128
        # x = self.blocks(x)  # 2 196 128 -> 2 16 384
        # x = self.stage(x)
        outs = []
        # print(self.size)
        for i, layer_name in enumerate(self.stages):
            x = layer_name(x)
            B, _, C = x.shape
            if i in self.out_indices:
                out = x.reshape(B, self.size[i], self.size[i], C).permute(0, 3, 1, 2)
                outs.append(out)
                # out = out.permute(0, 2, 3, 1).reshape(B, self.size[i] * self.size[i], C)
        return tuple(outs)


class Model(BaseModule):
    def __init__(self,
                 patch_size=16,
                 embed_dim=None,
                 num_heads=None,
                 key_dim=None,
                 depth=None,
                 attn_ratio=[2, 2, 2],
                 mlp_ratio=[2, 2, 2],
                 down_ops=None,
                 attention_activation=torch.nn.Hardswish,
                 mlp_activation=torch.nn.Hardswish,
                 hybrid_backbone=None,
                 num_classes=1000,
                 drop_path=0,
                 distillation=True
                 ):
        super(Model, self).__init__()
        self.backbone = LeViT(
            patch_size=patch_size,
            embed_dim=embed_dim,
            num_heads=num_heads,
            key_dim=key_dim,
            depth=depth,
            attn_ratio=attn_ratio,
            mlp_ratio=mlp_ratio,
            down_ops=down_ops,
            attention_activation=attention_activation,
            mlp_activation=mlp_activation,
            hybrid_backbone=hybrid_backbone,
            drop_path=drop_path,
        )
        self.head = LeViTClsHead(num_classes=num_classes, distillation=distillation, in_channels=embed_dim[-1])

    def forward(self, x):
        x = self.backbone(x)
        # B, C, W, H = x.shape
        # x = x.permute(0, 2, 1, 3).reshape(B, W * H, C)
        x = self.head(x)
        return x


def get_LeViT_model(params_name='LeViT_128S'):
    specification = {
        'LeViT_128S': {
            'C': '128_256_384', 'D': 16, 'N': '4_6_8', 'X': '2_3_4', 'drop_path': 0},
        'LeViT_128': {
            'C': '128_256_384', 'D': 16, 'N': '4_8_12', 'X': '4_4_4', 'drop_path': 0},
        'LeViT_192': {
            'C': '192_288_384', 'D': 32, 'N': '3_5_6', 'X': '4_4_4', 'drop_path': 0},
        'LeViT_256': {
            'C': '256_384_512', 'D': 32, 'N': '4_6_8', 'X': '4_4_4', 'drop_path': 0},
        'LeViT_384': {
            'C': '384_512_768', 'D': 32, 'N': '6_9_12', 'X': '4_4_4', 'drop_path': 0.1},
    }

    params = specification[params_name]

    C = params['C']  # embed_dim
    N = params['N']  # num_heads
    X = params['X']  # depth
    D = params['D']  # key_dim
    drop_path = params['drop_path']

    embed_dim = [int(x) for x in C.split('_')]
    num_heads = [int(x) for x in N.split('_')]
    depth = [int(x) for x in X.split('_')]
    act = torch.nn.Hardswish
    model = Model(
        patch_size=16,
        embed_dim=embed_dim,
        num_heads=num_heads,
        key_dim=[D] * 3,
        depth=depth,
        attn_ratio=[2, 2, 2],
        mlp_ratio=[2, 2, 2],
        down_ops=[
            # ('Subsample',key_dim, num_heads, attn_ratio, mlp_ratio, stride)
            ['Subsample', D, embed_dim[0] // D, 4, 2, 2],
            ['Subsample', D, embed_dim[1] // D, 4, 2, 2],
        ],
        attention_activation=act,
        mlp_activation=act,
        hybrid_backbone=hybrid_cnn(embed_dim[0], activation=act),
        num_classes=1000,
        drop_path=drop_path,
        distillation=True
    )
    return model
