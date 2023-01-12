# Copyright (c) OpenMMLab. All rights reserved.
# modified from
# https://github.com/huawei-noah/Efficient-AI-Backbones/tree/master/vig_pytorch
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import build_activation_layer, build_conv_layer, build_norm_layer
from mmcv.cnn.bricks import DropPath
from mmengine.model import Sequential

from mmcls.models.backbones.base_backbone import BaseBackbone
from mmcls.registry import MODELS


def get_2d_relative_pos_embed(embed_dim, grid_size):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, grid_size*grid_size]
    """
    pos_embed = get_2d_sincos_pos_embed(embed_dim, grid_size)
    relative_pos = 2 * np.matmul(pos_embed,
                                 pos_embed.transpose()) / pos_embed.shape[1]
    return relative_pos


def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or
    [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed],
                                   axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2,
                                              grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2,
                                              grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


def xy_pairwise_distance(x, y):
    """Compute pairwise distance of a point cloud.

    Args:
        x: tensor (batch_size, num_points, num_dims)
    Returns:
        pairwise distance: (batch_size, num_points, num_points)
    """
    with torch.no_grad():
        xy_inner = -2 * torch.matmul(x, y.transpose(2, 1))
        x_square = torch.sum(torch.mul(x, x), dim=-1, keepdim=True)
        y_square = torch.sum(torch.mul(y, y), dim=-1, keepdim=True)
        return x_square + xy_inner + y_square.transpose(2, 1)
        return xy_inner


# 在DenseDilatedKnnGraph中使用了归一化，所有特征相当于落在一个超球面上，这时两两特征之间的距离和夹角是一一对应的，
# 我们能否直接使用cos也就是点积代替距离呢？这样能节省一点计算量。但是测试结果有微小的差别，我认为是计算精度导致的。
# def xy_pairwise_distance(x, y):
#     with torch.no_grad():
#         xy_inner = -2 * torch.matmul(x, y.transpose(2, 1))
#         return xy_inner


def xy_dense_knn_matrix(x, y, k=16, relative_pos=None):
    """Get KNN based on the pairwise distance.

    Args:
        x: (batch_size, num_dims, num_points, 1)
        k: int
    Returns:
        nearest neighbors:
        (batch_size, num_points, k) (batch_size, num_points, k)
    """
    with torch.no_grad():
        x = x.transpose(2, 1).squeeze(-1)
        y = y.transpose(2, 1).squeeze(-1)
        batch_size, n_points, n_dims = x.shape
        dist = xy_pairwise_distance(x.detach(), y.detach())
        if relative_pos is not None:
            dist += relative_pos
        _, nn_idx = torch.topk(-dist, k=k)
        center_idx = torch.arange(
            0, n_points, device=x.device).repeat(batch_size, k,
                                                 1).transpose(2, 1)
    return torch.stack((nn_idx, center_idx), dim=0)


class DenseDilated(nn.Module):
    """Find dilated neighbor from neighbor list.

    edge_index: (2, batch_size, num_points, k)
    """

    def __init__(self, k=9, dilation=1, stochastic=False, epsilon=0.0):
        super(DenseDilated, self).__init__()
        self.dilation = dilation
        self.stochastic = stochastic
        self.epsilon = epsilon
        self.k = k

    def forward(self, edge_index):
        if self.stochastic:
            if torch.rand(1) < self.epsilon and self.training:
                num = self.k * self.dilation
                randnum = torch.randperm(num)[:self.k]
                edge_index = edge_index[:, :, :, randnum]
            else:
                edge_index = edge_index[:, :, :, ::self.dilation]
        else:
            edge_index = edge_index[:, :, :, ::self.dilation]
        return edge_index


class DenseDilatedKnnGraph(nn.Module):
    """Find the neighbors' indices based on dilated knn."""

    def __init__(self, k=9, dilation=1, stochastic=False, epsilon=0.0):
        super(DenseDilatedKnnGraph, self).__init__()
        self.dilation = dilation
        self.stochastic = stochastic
        self.epsilon = epsilon
        self.k = k
        self._dilated = DenseDilated(k, dilation, stochastic, epsilon)

    def forward(self, x, y=None, relative_pos=None):
        if y is not None:
            x = F.normalize(x, p=2.0, dim=1)
            y = F.normalize(y, p=2.0, dim=1)

            edge_index = xy_dense_knn_matrix(x, y, self.k * self.dilation,
                                             relative_pos)
        else:
            x = F.normalize(x, p=2.0, dim=1)
            y = x.clone()

            edge_index = xy_dense_knn_matrix(x, y, self.k * self.dilation,
                                             relative_pos)
        return self._dilated(edge_index)


class BasicConv(Sequential):

    def __init__(self, channels, act='relu', norm=None, bias=True, drop=0.):
        m = []
        for i in range(1, len(channels)):
            m.append(
                build_conv_layer(
                    None, channels[i - 1], channels[i], 1, bias=bias,
                    groups=4))
            if norm is not None:
                m.append(build_norm_layer(norm, channels[-1])[1])
            if act is not None:
                m.append(build_activation_layer(act))
            if drop > 0:
                m.append(nn.Dropout2d(drop))

        super(BasicConv, self).__init__(*m)


def batched_index_select(x, idx):
    r"""fetches neighbors features from a given neighbor idx

    Args:
        x (Tensor): input feature Tensor
                :math:
                `\mathbf{X} \in \mathbb{R}^{B \times C \times N \times 1}`.
        idx (Tensor): edge_idx
                :math:`\mathbf{X} \in \mathbb{R}^{B \times N \times l}`.
    Returns:
        Tensor: output neighbors features
            :math:`\mathbf{X} \in \mathbb{R}^{B \times C \times N \times k}`.
    """
    batch_size, num_dims, num_vertices_reduced = x.shape[:3]
    _, num_vertices, k = idx.shape
    idx_base = torch.arange(
        0, batch_size, device=idx.device).view(-1, 1, 1) * num_vertices_reduced
    idx = idx + idx_base
    idx = idx.contiguous().view(-1)

    x = x.transpose(2, 1)
    feature = x.contiguous().view(batch_size * num_vertices_reduced,
                                  -1)[idx, :]
    feature = feature.view(batch_size, num_vertices, k,
                           num_dims).permute(0, 3, 1, 2).contiguous()
    return feature


class MRConv2d(nn.Module):
    """Max-Relative Graph Convolution (Paper: https://arxiv.org/abs/1904.03751)
    for dense data type."""

    def __init__(self, in_channels, out_channels, act, norm=None, bias=True):
        super(MRConv2d, self).__init__()
        self.nn = BasicConv([in_channels * 2, out_channels], act, norm, bias)

    def forward(self, x, edge_index, y=None):
        x_i = batched_index_select(x, edge_index[1])
        if y is not None:
            x_j = batched_index_select(y, edge_index[0])
        else:
            x_j = batched_index_select(x, edge_index[0])
        x_j, _ = torch.max(x_j - x_i, -1, keepdim=True)
        b, c, n, _ = x.shape
        x = torch.cat([x.unsqueeze(2), x_j.unsqueeze(2)],
                      dim=2).reshape(b, 2 * c, n, _)
        return self.nn(x)


class EdgeConv2d(nn.Module):
    """Edge convolution layer (with activation, batch normalization) for dense
    data type."""

    def __init__(self, in_channels, out_channels, act, norm=None, bias=True):
        super(EdgeConv2d, self).__init__()
        self.nn = BasicConv([in_channels * 2, out_channels], act, norm, bias)

    def forward(self, x, edge_index, y=None):
        x_i = batched_index_select(x, edge_index[1])
        if y is not None:
            x_j = batched_index_select(y, edge_index[0])
        else:
            x_j = batched_index_select(x, edge_index[0])
        max_value, _ = torch.max(
            self.nn(torch.cat([x_i, x_j - x_i], dim=1)), -1, keepdim=True)
        return max_value


class GraphSAGE(nn.Module):
    """GraphSAGE Graph Convolution (Paper: https://arxiv.org/abs/1706.02216)
    for dense data type."""

    def __init__(self, in_channels, out_channels, act, norm=None, bias=True):
        super(GraphSAGE, self).__init__()
        self.nn1 = BasicConv([in_channels, in_channels], act, norm, bias)
        self.nn2 = BasicConv([in_channels * 2, out_channels], act, norm, bias)

    def forward(self, x, edge_index, y=None):
        if y is not None:
            x_j = batched_index_select(y, edge_index[0])
        else:
            x_j = batched_index_select(x, edge_index[0])
        x_j, _ = torch.max(self.nn1(x_j), -1, keepdim=True)
        return self.nn2(torch.cat([x, x_j], dim=1))


class GINConv2d(nn.Module):
    """GIN Graph Convolution (Paper: https://arxiv.org/abs/1810.00826) for
    dense data type."""

    def __init__(self, in_channels, out_channels, act, norm=None, bias=True):
        super(GINConv2d, self).__init__()
        self.nn = BasicConv([in_channels, out_channels], act, norm, bias)
        eps_init = 0.0
        self.eps = nn.Parameter(torch.Tensor([eps_init]))

    def forward(self, x, edge_index, y=None):
        if y is not None:
            x_j = batched_index_select(y, edge_index[0])
        else:
            x_j = batched_index_select(x, edge_index[0])
        x_j = torch.sum(x_j, -1, keepdim=True)
        return self.nn((1 + self.eps) * x + x_j)


class GraphConv2d(nn.Module):
    """Static graph convolution layer."""

    def __init__(self,
                 in_channels,
                 out_channels,
                 conv,
                 act,
                 norm=None,
                 bias=True):
        super(GraphConv2d, self).__init__()
        if conv == 'edge':
            self.gconv = EdgeConv2d(in_channels, out_channels, act, norm, bias)
        elif conv == 'mr':
            self.gconv = MRConv2d(in_channels, out_channels, act, norm, bias)
        elif conv == 'sage':
            self.gconv = GraphSAGE(in_channels, out_channels, act, norm, bias)
        elif conv == 'gin':
            self.gconv = GINConv2d(in_channels, out_channels, act, norm, bias)
        else:
            raise NotImplementedError(
                'graph_conv_type:{} is not supported'.format(conv))

    def forward(self, x, edge_index, y=None):
        return self.gconv(x, edge_index, y)


class DyGraphConv2d(GraphConv2d):
    """Dynamic graph convolution layer."""

    def __init__(self,
                 in_channels,
                 out_channels,
                 k=9,
                 dilation=1,
                 conv='edge',
                 act=dict(type='GELU'),
                 norm=None,
                 bias=True,
                 stochastic=False,
                 epsilon=0.0,
                 r=1):
        super(DyGraphConv2d, self).__init__(in_channels, out_channels, conv,
                                            act, norm, bias)
        self.k = k
        self.d = dilation
        self.r = r
        self.dilated_knn_graph = DenseDilatedKnnGraph(k, dilation, stochastic,
                                                      epsilon)

    def forward(self, x, relative_pos=None):
        B, C, H, W = x.shape
        y = None
        if self.r > 1:
            y = F.avg_pool2d(x, self.r, self.r)
            y = y.reshape(B, C, -1, 1).contiguous()
        x = x.reshape(B, C, -1, 1).contiguous()
        edge_index = self.dilated_knn_graph(x, y, relative_pos)
        x = super(DyGraphConv2d, self).forward(x, edge_index, y)
        return x.reshape(B, -1, H, W).contiguous()


class Grapher(nn.Module):
    """Grapher module with graph convolution and fc layers."""

    def __init__(self,
                 in_channels,
                 k=9,
                 dilation=1,
                 conv='mr',
                 act=dict(type='GELU'),
                 norm=None,
                 bias=True,
                 stochastic=False,
                 epsilon=0.0,
                 r=1,
                 n=196,
                 drop_path=0.0,
                 relative_pos=False):
        super(Grapher, self).__init__()
        self.channels = in_channels
        self.n = n
        self.r = r
        self.fc1 = Sequential(
            build_conv_layer(
                None, in_channels, in_channels, 1, stride=1, padding=0),
            build_norm_layer(dict(type='BN'), in_channels)[1],
        )
        self.graph_conv = DyGraphConv2d(in_channels, in_channels * 2, k,
                                        dilation, conv, act, norm, bias,
                                        stochastic, epsilon, r)
        self.fc2 = Sequential(
            build_conv_layer(
                None, in_channels * 2, in_channels, 1, stride=1, padding=0),
            build_norm_layer(dict(type='BN'), in_channels)[1],
        )
        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()

        self.relative_pos = None
        if relative_pos:
            print('using relative_pos')
            relative_pos_tensor = torch.from_numpy(
                np.float32(
                    get_2d_relative_pos_embed(in_channels, int(
                        n**0.5)))).unsqueeze(0).unsqueeze(1)
            relative_pos_tensor = F.interpolate(
                relative_pos_tensor,
                size=(n, n // (r * r)),
                mode='bicubic',
                align_corners=False)
            self.relative_pos = nn.Parameter(
                -relative_pos_tensor.squeeze(1), requires_grad=False)

    def _get_relative_pos(self, relative_pos, H, W):
        if relative_pos is None or H * W == self.n:
            return relative_pos
        else:
            N = H * W
            N_reduced = N // (self.r * self.r)
            return F.interpolate(
                relative_pos.unsqueeze(0), size=(N, N_reduced),
                mode='bicubic').squeeze(0)

    def forward(self, x):
        B, C, H, W = x.shape
        relative_pos = self._get_relative_pos(self.relative_pos, H, W)
        shortcut = x
        x = self.fc1(x)
        x = self.graph_conv(x, relative_pos)
        x = self.fc2(x)
        x = self.drop_path(x) + shortcut
        return x


class FFN(nn.Module):
    """"out_features = out_features or in_features\n
        hidden_features = hidden_features or in_features"""

    def __init__(self,
                 in_features,
                 hidden_features=None,
                 out_features=None,
                 act=dict(type='GELU'),
                 drop_path=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = Sequential(
            build_conv_layer(
                None, in_features, hidden_features, 1, stride=1, padding=0),
            build_norm_layer(dict(type='BN'), hidden_features)[1],
        )
        self.act = build_activation_layer(act)
        self.fc2 = Sequential(
            build_conv_layer(
                None, hidden_features, out_features, 1, stride=1, padding=0),
            build_norm_layer(dict(type='BN'), out_features)[1],
        )
        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        shortcut = x
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.drop_path(x) + shortcut
        return x


class Stem(nn.Module):
    """ Image to Visual Word Embedding
    Overlap: https://arxiv.org/pdf/2106.13797.pdf
    """

    def __init__(self, in_dim=3, out_dim=768, act='relu'):
        super().__init__()
        self.convs = Sequential(
            build_conv_layer(
                None, in_dim, out_dim // 8, 3, stride=2, padding=1),
            build_norm_layer(dict(type='BN'), out_dim // 8)[1],
            build_activation_layer(act),
            build_conv_layer(
                None, out_dim // 8, out_dim // 4, 3, stride=2, padding=1),
            build_norm_layer(dict(type='BN'), out_dim // 4)[1],
            build_activation_layer(act),
            build_conv_layer(
                None, out_dim // 4, out_dim // 2, 3, stride=2, padding=1),
            build_norm_layer(dict(type='BN'), out_dim // 2)[1],
            build_activation_layer(act),
            build_conv_layer(
                None, out_dim // 2, out_dim, 3, stride=2, padding=1),
            build_norm_layer(dict(type='BN'), out_dim)[1],
            build_activation_layer(act),
            build_conv_layer(None, out_dim, out_dim, 3, stride=1, padding=1),
            build_norm_layer(dict(type='BN'), out_dim)[1],
        )

    def forward(self, x):
        x = self.convs(x)
        return x


@MODELS.register_module()
class Vig(BaseBackbone):
    # n_blocks,channels
    arch_settings = {'tiny': [12, 192], 'small': [16, 320], 'base': [16, 640]}

    def __init__(self,
                 arch,
                 k=9,
                 act_cfg=dict(type='GELU'),
                 norm_cfg=dict(type='BN'),
                 graph_conv_bias=True,
                 graph_conv_type='mr',
                 epsilon=0.2,
                 use_dilation=True,
                 use_stochastic=False,
                 drop_path=0.,
                 relative_pos=False,
                 norm_eval=False,
                 frozen_stages=0,
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        arch = self.arch_settings[arch]
        self.n_blocks = arch[0]
        channels = arch[1]
        self.stem = Stem(out_dim=channels, act=act_cfg)
        dpr = [x.item() for x in torch.linspace(0, drop_path, self.n_blocks)
               ]  # stochastic depth decay rule
        num_knn = [
            int(x.item()) for x in torch.linspace(k, 2 * k, self.n_blocks)
        ]  # number of knn's k
        max_dilation = 196 // max(num_knn)
        self.norm_eval = norm_eval
        self.pos_embed = nn.Parameter(torch.zeros(1, channels, 14, 14))
        self.frozen_stages = frozen_stages
        # i在循环里面，无法直接拿出来
        # dilation = min(i // 4 + 1, max_dilation) if use_dilation else 1
        self.stage_blocks = Sequential(*[
            Sequential(
                Grapher(
                    in_channels=channels,
                    k=num_knn[i],
                    dilation=min(i // 4 +
                                 1, max_dilation) if use_dilation else 1,
                    conv=graph_conv_type,
                    act=act_cfg,
                    norm=norm_cfg,
                    bias=graph_conv_bias,
                    stochastic=use_stochastic,
                    epsilon=epsilon,
                    drop_path=dpr[i],
                    relative_pos=relative_pos),
                FFN(in_features=channels,
                    hidden_features=channels * 4,
                    act=act_cfg,
                    drop_path=dpr[i])) for i in range(self.n_blocks)
        ])

    def forward(self, inputs):
        outs = []
        x = self.stem(inputs) + self.pos_embed

        for i in range(self.n_blocks):
            x = self.stage_blocks[i](x)
            outs.append(x)

        x = F.adaptive_avg_pool2d(x, 1)
        outs.append(x)
        return outs

    def _freeze_stages(self):
        self.stem.eval()
        for i in range(self.frozen_stages):
            m = self.stage_blocks[i]
            m.eval()
            for param in m.parameters():
                param.requires_grad = False

    def train(self, mode=True):
        super(Vig, self).train(mode)
        self._freeze_stages()
        if mode and self.norm_eval:
            for m in self.modules():
                # trick: eval have effect on BatchNorm only
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()