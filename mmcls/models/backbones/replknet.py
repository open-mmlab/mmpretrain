# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
import sys
import os
from mmcv.cnn.bricks import DropPath
from mmcv.cnn import build_activation_layer, build_conv_layer, build_norm_layer
from mmengine.model import BaseModule, Sequential
from mmengine.utils.dl_utils.parrots_wrapper import _BatchNorm

from mmcls.registry import MODELS
from .base_backbone import BaseBackbone


def get_conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias):
    if type(kernel_size) is int:
        use_large_impl = kernel_size > 5
    else:
        assert len(kernel_size) == 2 and kernel_size[0] == kernel_size[1]
        use_large_impl = kernel_size[0] > 5
    has_large_impl = 'LARGE_KERNEL_CONV_IMPL' in os.environ
    if has_large_impl and in_channels == out_channels and out_channels == groups and use_large_impl and stride == 1 and padding == kernel_size // 2 and dilation == 1:
        sys.path.append(os.environ['LARGE_KERNEL_CONV_IMPL'])
        #   Please follow the instructions https://github.com/DingXiaoH/RepLKNet-pytorch/blob/main/README.md
        #   export LARGE_KERNEL_CONV_IMPL=absolute_path_to_where_you_cloned_the_example (i.e., depthwise_conv2d_implicit_gemm.py)
        # TODO more efficient PyTorch implementations of large-kernel convolutions. Pull requests are welcomed.
        # Or you may try MegEngine. We have integrated an efficient implementation into MegEngine and it will automatically use it.
        from depthwise_conv2d_implicit_gemm import DepthWiseConv2dImplicitGEMM
        return DepthWiseConv2dImplicitGEMM(in_channels, kernel_size, bias=bias)
    else:
        return nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                         padding=padding, dilation=dilation, groups=groups, bias=bias)

use_sync_bn = False

def enable_sync_bn():
    global use_sync_bn
    use_sync_bn = True

def get_bn(channels):
    if use_sync_bn:
        return nn.SyncBatchNorm(channels)
    else:
        return nn.BatchNorm2d(channels)

def conv_bn(in_channels, out_channels, kernel_size, stride, padding, groups, dilation=1):
    if padding is None:
        padding = kernel_size // 2
    result = nn.Sequential()
    result.add_module('conv', get_conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                         stride=stride, padding=padding, dilation=dilation, groups=groups, bias=False))
    result.add_module('bn', get_bn(out_channels))
    return result

def conv_bn_relu(in_channels, out_channels, kernel_size, stride, padding, groups, dilation=1):
    if padding is None:
        padding = kernel_size // 2
    result = conv_bn(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                         stride=stride, padding=padding, groups=groups, dilation=dilation)
    result.add_module('nonlinear', nn.ReLU())
    return result

def fuse_bn(conv, bn):
    kernel = conv.weight
    running_mean = bn.running_mean
    running_var = bn.running_var
    gamma = bn.weight
    beta = bn.bias
    eps = bn.eps
    std = (running_var + eps).sqrt()
    t = (gamma / std).reshape(-1, 1, 1, 1)
    return kernel * t, beta - running_mean * gamma / std


class ReparamLargeKernelConv(BaseModule):

    def __init__(self, 
                 in_channels, 
                 out_channels, 
                 kernel_size,
                 stride, groups,
                 small_kernel,
                 small_kernel_merged=False,
                 init_cfg=None):
        super(ReparamLargeKernelConv, self).__init__(init_cfg)
        self.kernel_size = kernel_size
        self.small_kernel = small_kernel
        # We assume the conv does not change the feature map size, so padding = k//2. Otherwise, you may configure padding as you wish, and change the padding of small_conv accordingly.
        padding = kernel_size // 2
        if small_kernel_merged:
            self.lkb_reparam = get_conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                          stride=stride, padding=padding, dilation=1, groups=groups, bias=True)
        else:
            self.lkb_origin = conv_bn(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                      stride=stride, padding=padding, dilation=1, groups=groups)
            if small_kernel is not None:
                assert small_kernel <= kernel_size, 'The kernel size for re-param cannot be larger than the large kernel!'
                self.small_conv = conv_bn(in_channels=in_channels, out_channels=out_channels, kernel_size=small_kernel,
                                             stride=stride, padding=small_kernel//2, groups=groups, dilation=1)

    def forward(self, inputs):
        if hasattr(self, 'lkb_reparam'):
            out = self.lkb_reparam(inputs)
        else:
            out = self.lkb_origin(inputs)
            if hasattr(self, 'small_conv'):
                out += self.small_conv(inputs)
        return out

    def get_equivalent_kernel_bias(self):
        eq_k, eq_b = fuse_bn(self.lkb_origin.conv, self.lkb_origin.bn)
        if hasattr(self, 'small_conv'):
            small_k, small_b = fuse_bn(self.small_conv.conv, self.small_conv.bn)
            eq_b += small_b
            #   add to the central part
            eq_k += nn.functional.pad(small_k, [(self.kernel_size - self.small_kernel) // 2] * 4)
        return eq_k, eq_b

    def merge_kernel(self):
        eq_k, eq_b = self.get_equivalent_kernel_bias()
        self.lkb_reparam = get_conv2d(in_channels=self.lkb_origin.conv.in_channels,
                                     out_channels=self.lkb_origin.conv.out_channels,
                                     kernel_size=self.lkb_origin.conv.kernel_size, stride=self.lkb_origin.conv.stride,
                                     padding=self.lkb_origin.conv.padding, dilation=self.lkb_origin.conv.dilation,
                                     groups=self.lkb_origin.conv.groups, bias=True)
        self.lkb_reparam.weight.data = eq_k
        self.lkb_reparam.bias.data = eq_b
        self.__delattr__('lkb_origin')
        if hasattr(self, 'small_conv'):
            self.__delattr__('small_conv')


class ConvFFN(BaseModule):

    def __init__(self, 
                 in_channels, 
                 internal_channels, 
                 out_channels, 
                 drop_path,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='GELU'),
                 init_cfg=None):
        super(ConvFFN, self).__init__(init_cfg)
        self.drop_path = DropPath(drop_prob=drop_path) if drop_path > 0. else nn.Identity()
        self.preffn_bn = build_norm_layer(norm_cfg, in_channels)[1]
        self.pw1 = conv_bn(in_channels=in_channels, out_channels=internal_channels, kernel_size=1, stride=1, padding=0, groups=1)
        self.pw2 = conv_bn(in_channels=internal_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0, groups=1)
        self.nonlinear = build_activation_layer(act_cfg)

    def forward(self, x):
        out = self.preffn_bn(x)
        out = self.pw1(out)
        out = self.nonlinear(out)
        out = self.pw2(out)
        return x + self.drop_path(out)


class RepLKBlock(BaseModule):

    def __init__(self, 
                 in_channels, 
                 dw_channels, 
                 block_lk_size, 
                 small_kernel, 
                 drop_path, 
                 small_kernel_merged=False, 
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='ReLU'),
                 init_cfg=None):
        super(RepLKBlock, self).__init__(init_cfg)
        self.pw1 = conv_bn_relu(in_channels, dw_channels, 1, 1, 0, groups=1)
        self.pw2 = conv_bn(dw_channels, in_channels, 1, 1, 0, groups=1)
        self.large_kernel = ReparamLargeKernelConv(in_channels=dw_channels, out_channels=dw_channels, kernel_size=block_lk_size,
                                                  stride=1, groups=dw_channels, small_kernel=small_kernel, small_kernel_merged=small_kernel_merged)
        self.lk_nonlinear = build_activation_layer(act_cfg)
        self.prelkb_bn = build_norm_layer(norm_cfg, in_channels)[1]
        self.drop_path = DropPath(drop_prob=drop_path) if drop_path > 0. else nn.Identity()
        print('drop path:', self.drop_path)

    def forward(self, x):
        out = self.prelkb_bn(x)
        out = self.pw1(out)
        out = self.large_kernel(out)
        out = self.lk_nonlinear(out)
        out = self.pw2(out)
        return x + self.drop_path(out)


class RepLKNetStage(BaseModule):

    def __init__(self, 
                 channels, 
                 num_blocks, 
                 stage_lk_size, 
                 drop_path,
                 small_kernel, 
                 dw_ratio=1, 
                 ffn_ratio=4,
                 with_cp=False,      # train with torch.utils.checkpoint to save memory
                 small_kernel_merged=False,
                 norm_intermediate_features=False,
                 norm_cfg=dict(type='BN'),
                 init_cfg=None):
        super(RepLKNetStage, self).__init__(init_cfg)
        self.with_cp = with_cp
        blks = []
        for i in range(num_blocks):
            block_drop_path = drop_path[i] if isinstance(drop_path, list) else drop_path
            #   Assume all RepLK Blocks within a stage share the same lk_size. You may tune it on your own model.
            replk_block = RepLKBlock(in_channels=channels, dw_channels=int(channels * dw_ratio), block_lk_size=stage_lk_size,
                                     small_kernel=small_kernel, drop_path=block_drop_path, small_kernel_merged=small_kernel_merged)
            convffn_block = ConvFFN(in_channels=channels, internal_channels=int(channels * ffn_ratio), out_channels=channels,
                                    drop_path=block_drop_path)
            blks.append(replk_block)
            blks.append(convffn_block)
        self.blocks = nn.ModuleList(blks)
        if norm_intermediate_features:
            self.norm = build_norm_layer(norm_cfg, channels)[1]    #   Only use this with RepLKNet-XL on downstream tasks
        else:
            self.norm = nn.Identity()

    def forward(self, x):
        for blk in self.blocks:
            if self.with_cp:
                x = checkpoint.checkpoint(blk, x)   # Save training memory
            else:
                x = blk(x)
        return x


@MODELS.register_module()
class RepLKNet(BaseBackbone):

    arch_settings = {
        '31B':
        dict(
            large_kernel_sizes=[31, 29, 27, 13],
            layers=[2, 2, 18, 2],
            channels=[128, 256, 512, 1024],
            small_kernel=5,
            dw_ratio=1),
        '31L':
        dict(
            large_kernel_sizes=[31, 29, 27, 13],
            layers=[2, 2, 18, 2],
            channels=[192, 384, 768, 1536],
            small_kernel=5,
            dw_ratio=1),
        'XL':
        dict(
            large_kernel_sizes=[27, 27, 27, 13],
            layers=[2, 2, 18, 2],
            channels=[256, 512, 1024, 2048],
            small_kernel=None,
            dw_ratio=1.5),
    }

    def __init__(self,
                 arch,
                 in_channels=3,
                 ffn_ratio=4,
                 out_indices=(3, ),
                 strides=(2, 2, 2, 2),
                 dilations=(1, 1, 1, 1),
                 frozen_stages=-1,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='ReLU'),
                 with_cp=False,
                 drop_path_rate=0.3,
                 small_kernel_merged=False,
                 norm_intermediate_features=False,
                 norm_eval=False,
                 init_cfg=[
                     dict(type='Kaiming', layer=['Conv2d']),
                     dict(
                         type='Constant',
                         val=1,
                         layer=['_BatchNorm', 'GroupNorm'])
                 ]):
        super(RepLKNet, self).__init__(init_cfg)

        if isinstance(arch, str):
            assert arch in self.arch_settings, \
                f'"arch": "{arch}" is not one of the arch_settings'
            arch = self.arch_settings[arch]
        elif not isinstance(arch, dict):
            raise TypeError('Expect "arch" to be either a string '
                            f'or a dict, got {type(arch)}')

        assert len(arch['layers']) == len(
            arch['channels']) == len(strides) == len(dilations)
        assert max(out_indices) < len(arch['layers'])

        self.arch = arch
        self.in_channels = in_channels
        self.out_indices = out_indices
        self.strides = strides
        self.dilations = dilations
        self.frozen_stages = frozen_stages
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.with_cp = with_cp
        self.drop_path_rate = drop_path_rate
        self.small_kernel_merged = small_kernel_merged
        self.norm_eval = norm_eval
        self.use_sync_bn = use_sync_bn
        self.norm_intermediate_features = norm_intermediate_features

        self.out_indices = out_indices
        if use_sync_bn:
            enable_sync_bn()

        base_width = self.arch['channels'][0]
        self.norm_intermediate_features = norm_intermediate_features
        self.num_stages = len(self.arch['layers'])
        self.stem = nn.ModuleList([
            conv_bn_relu(in_channels=in_channels, out_channels=base_width, kernel_size=3, stride=2, padding=1, groups=1),
            conv_bn_relu(in_channels=base_width, out_channels=base_width, kernel_size=3, stride=1, padding=1, groups=base_width),
            conv_bn_relu(in_channels=base_width, out_channels=base_width, kernel_size=1, stride=1, padding=0, groups=1),
            conv_bn_relu(in_channels=base_width, out_channels=base_width, kernel_size=3, stride=2, padding=1, groups=base_width)])
        # stochastic depth. We set block-wise drop-path rate. The higher level blocks are more likely to be dropped. This implementation follows Swin.
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(self.arch['layers']))]
        self.stages = nn.ModuleList()
        self.transitions = nn.ModuleList()
        for stage_idx in range(self.num_stages):
            layer = RepLKNetStage(channels=self.arch['channels'][stage_idx], num_blocks=self.arch['layers'][stage_idx],
                                  stage_lk_size=self.arch['large_kernel_sizes'][stage_idx],
                                  drop_path=dpr[sum(self.arch['layers'][:stage_idx]):sum(self.arch['layers'][:stage_idx + 1])],
                                  small_kernel=self.arch['small_kernel'], dw_ratio=self.arch['dw_ratio'], ffn_ratio=ffn_ratio,
                                  with_cp=with_cp, small_kernel_merged=small_kernel_merged,
                                  norm_intermediate_features=(stage_idx in out_indices))
            self.stages.append(layer)
            if stage_idx < len(self.arch['layers']) - 1:
                transition = nn.Sequential(
                    conv_bn_relu(self.arch['channels'][stage_idx], self.arch['channels'][stage_idx + 1], 1, 1, 0, groups=1),
                    conv_bn_relu(self.arch['channels'][stage_idx + 1], self.arch['channels'][stage_idx + 1], 3, stride=2, padding=1, groups=self.arch['channels'][stage_idx + 1]))
                self.transitions.append(transition)


    def forward_features(self, x):
        x = self.stem[0](x)
        for stem_layer in self.stem[1:]:
            if self.with_cp:
                x = checkpoint.checkpoint(stem_layer, x)     # save memory
            else:
                x = stem_layer(x)

        #   Need the intermediate feature maps
        outs = []
        for stage_idx in range(self.num_stages):
            x = self.stages[stage_idx](x)
            if stage_idx in self.out_indices:
                outs.append(self.stages[stage_idx].norm(x))     # For RepLKNet-XL normalize the features before feeding them into the heads
            if stage_idx < self.num_stages - 1:
                x = self.transitions[stage_idx](x)
        return outs


    def forward(self, x):
        x = self.forward_features(x)
        return tuple(x)

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            self.stem.eval()
            for param in self.stem.parameters():
                param.requires_grad = False
        for i in range(self.frozen_stages):
            stage = getattr(self, f'stages.{i}')
            stage.eval()
            for param in stage.parameters():
                param.requires_grad = False

    def train(self, mode=True):
        super(RepLKNet, self).train(mode)
        self._freeze_stages()
        if mode and self.norm_eval:
            for m in self.modules():
                if isinstance(m, _BatchNorm):
                    m.eval()

    def switch_to_deploy(self):
        for m in self.modules():
            if hasattr(m, 'merge_kernel'):
                m.merge_kernel()
        self.small_kernel_merged = True

    #   If your framework cannot automatically fuse BN for inference, you may do it manually.
    #   The BNs after and before conv layers can be removed.
    #   No need to call this if your framework support automatic BN fusion.
    def deep_fuse_BN(self):
        for m in self.modules():
            if not isinstance(m, nn.Sequential):
                continue
            if not len(m) in [2, 3]:  # Only handle conv-BN or conv-BN-relu
                continue
            #   If you use a custom Conv2d impl, assume it also has 'kernel_size' and 'weight'
            if hasattr(m[0], 'kernel_size') and hasattr(m[0], 'weight') and isinstance(m[1], nn.BatchNorm2d):
                conv = m[0]
                bn = m[1]
                fused_kernel, fused_bias = fuse_bn(conv, bn)
                fused_conv = get_conv2d(conv.in_channels, conv.out_channels, kernel_size=conv.kernel_size,
                                        stride=conv.stride,
                                        padding=conv.padding, dilation=conv.dilation, groups=conv.groups, bias=True)
                fused_conv.weight.data = fused_kernel
                fused_conv.bias.data = fused_bias
                m[0] = fused_conv
                m[1] = nn.Identity()


# def create_RepLKNet31B(drop_path_rate=0.3, num_classes=1000, with_cp=True, small_kernel_merged=False):
#     return RepLKNet(arch='31B')

# def create_RepLKNet31L(drop_path_rate=0.3, num_classes=1000, with_cp=True, small_kernel_merged=False):
#     return RepLKNet(large_kernel_sizes=[31,29,27,13], layers=[2,2,18,2], channels=[192,384,768,1536],
#                     drop_path_rate=drop_path_rate, small_kernel=5, num_classes=num_classes, with_cp=with_cp,
#                     small_kernel_merged=small_kernel_merged)

# def create_RepLKNetXL(drop_path_rate=0.3, num_classes=1000, with_cp=True, small_kernel_merged=False):
#     return RepLKNet(large_kernel_sizes=[27,27,27,13], layers=[2,2,18,2], channels=[256,512,1024,2048],
#                     drop_path_rate=drop_path_rate, small_kernel=None, dw_ratio=1.5,
#                     num_classes=num_classes, with_cp=with_cp,
#                     small_kernel_merged=small_kernel_merged)

# if __name__ == '__main__':
#     model = create_RepLKNet31B(small_kernel_merged=False)
#     model.eval()
#     print('------------------- training-time model -------------')
#     for i in model.state_dict().keys():
#         print(i)
#     print(model)
    # x = torch.randn(2, 3, 224, 224)
    # origin_y = model(x)
    # model.switch_to_deploy()
    # print('------------------- after re-param -------------')
    # print(model)
    # reparam_y = model(x)
    # print('------------------- the difference is ------------------------')
    # print((origin_y - reparam_y).abs().sum())




