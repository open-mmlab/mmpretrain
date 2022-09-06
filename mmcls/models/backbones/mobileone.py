# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import build_activation_layer, build_conv_layer, build_norm_layer
from mmcv.runner import BaseModule, ModuleList, Sequential
from mmcv.utils.parrots_wrapper import _BatchNorm

from ..builder import BACKBONES
from ..utils.se_layer import SELayer
from .base_backbone import BaseBackbone


class MobileOneBlock(BaseModule):
    """MobileOne block for MobileOne backbone.

    Args:
        in_channels (int): The input channels of the block.
        out_channels (int): The output channels of the block.
        stride (int): Stride of the 3x3 and 1x1 convolution layer. Default: 1.
        padding (int): Padding of the 3x3 convolution layer.
        se_cfg (None or dict): The configuration of the se module.
            Default: None.
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed. Default: False.
        norm_cfg (dict): dictionary to construct and config norm layer.
            Default: dict(type='BN', requires_grad=True).
        act_cfg (dict): Config dict for activation layer.
            Default: dict(type='ReLU').
        deploy (bool): Whether to switch the model structure to
            deployment mode. Default: False.
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 num_conv_branches,
                 stride=1,
                 padding=1,
                 dilation: int = 1,
                 groups=1,
                 se_cfg=None,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='ReLU'),
                 deploy=False,
                 init_cfg=None):
        super(MobileOneBlock, self).__init__(init_cfg)

        assert se_cfg is None or isinstance(se_cfg, dict), se_cfg
        if se_cfg is not None:
            self.se = SELayer(channels=out_channels, **se_cfg)
        else:
            self.se = nn.Identity()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.num_conv_branches = num_conv_branches
        self.stride = stride
        self.padding = padding
        self.se_cfg = se_cfg
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.deploy = deploy
        self.groups = groups
        self.dilation = dilation

        if deploy:
            self.branch_reparam = build_conv_layer(
                conv_cfg,
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                groups=self.groups,
                stride=stride,
                padding=padding,
                dilation=dilation,
                bias=True)
        else:
            # judge if input shape and output shape are the same.
            # If true, add a normalized identity shortcut.
            if out_channels == in_channels and stride == 1:
                self.branch_norm = build_norm_layer(norm_cfg, in_channels)[1]
            else:
                self.branch_norm = None

            self.branch_1x1 = None
            if kernel_size > 1:
                self.branch_1x1 = self.create_conv_bn(kernel_size=1)

            self.branch_3x3_list = ModuleList()
            for _ in range(num_conv_branches):
                self.branch_3x3_list.append(
                    self.create_conv_bn(
                        kernel_size=kernel_size, padding=padding))

        self.act = build_activation_layer(act_cfg)

    def create_conv_bn(self, kernel_size, dilation=1, padding=0):
        conv_bn = Sequential()
        conv_bn.add_module(
            'conv',
            build_conv_layer(
                self.conv_cfg,
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                kernel_size=kernel_size,
                groups=self.groups,
                stride=self.stride,
                dilation=dilation,
                padding=padding,
                bias=False))
        conv_bn.add_module(
            'norm',
            build_norm_layer(self.norm_cfg, num_features=self.out_channels)[1])

        return conv_bn

    def forward(self, x):

        def _inner_forward(inputs):
            if self.deploy:
                return self.branch_reparam(inputs)

            if self.branch_norm is None:
                inner_out = 0
            else:
                inner_out = self.branch_norm(inputs)

            if self.branch_1x1 is not None:
                inner_out += self.branch_1x1(inputs)

            for branch_3x3 in self.branch_3x3_list:
                inner_out += branch_3x3(inputs)

            return inner_out

        out = _inner_forward(x)
        out = self.se(out)
        out = self.act(out)

        return out

    def switch_to_deploy(self):
        """Switch the model structure from training mode to deployment mode."""
        if self.deploy:
            return
        assert self.norm_cfg['type'] == 'BN', \
            "Switch is not allowed when norm_cfg['type'] != 'BN'."

        reparam_weight, reparam_bias = self.reparameterize()
        self.branch_reparam = build_conv_layer(
            self.conv_cfg,
            self.in_channels,
            self.out_channels,
            kernel_size=3,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
            bias=True)
        self.branch_reparam.weight.data = reparam_weight
        self.branch_reparam.bias.data = reparam_bias

        for param in self.parameters():
            param.detach_()
        delattr(self, 'branch_3x3_list')
        delattr(self, 'branch_1x1')
        delattr(self, 'branch_norm')

        self.deploy = True

    def reparameterize(self):
        """Fuse all the parameters of all branches.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Parameters after fusion of all
                branches. the first element is the weights and the second is
                the bias.
        """
        weight_3x3, bias_3x3 = 0, 0
        for branch_3x3 in self.branch_3x3_list:
            weight, bias = self._fuse_conv_bn(branch_3x3)
            weight_3x3 += weight
            bias_3x3 += bias

        weight_1x1, bias_1x1 = 0, 0
        if self.branch_1x1 is not None:
            weight_1x1, bias_1x1 = self._fuse_conv_bn(self.branch_1x1)
            # Pad scale branch kernel to match conv branch kernel size.
            pad = self.kernel_size // 2
            weight_1x1 = F.pad(weight_1x1, [pad, pad, pad, pad])

        weight_norm, bias_norm = 0, 0
        if self.branch_norm:
            tmp_conv_bn = self._norm_to_conv(self.branch_norm)
            weight_norm, bias_norm = self._fuse_conv_bn(tmp_conv_bn)

        return (weight_3x3 + weight_1x1 + weight_norm,
                bias_3x3 + bias_1x1 + bias_norm)

    def _fuse_conv_bn(self, branch):
        """Fuse the parameters in a branch with a conv and bn.

        Args:
            branch (mmcv.runner.Sequential): A branch with conv and bn.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: The parameters obtained after
                fusing the parameters of conv and bn in one branch.
                The first element is the weight and the second is the bias.
        """
        if branch is None:
            return 0, 0
        kernel = branch.conv.weight
        running_mean = branch.norm.running_mean
        running_var = branch.norm.running_var
        gamma = branch.norm.weight
        beta = branch.norm.bias
        eps = branch.norm.eps

        std = (running_var + eps).sqrt()
        fused_weight = (gamma / std).reshape(-1, 1, 1, 1) * kernel
        fused_bias = beta - running_mean * gamma / std

        return fused_weight, fused_bias

    def _norm_to_conv(self, branch_nrom):
        """Convert a norm layer to a conv-bn sequence.

        Args:
            branch (nn.BatchNorm2d): A branch only with bn in the block.

        Returns:
            tmp_conv3x3 (mmcv.runner.Sequential): a sequential with conv3x3 and
                bn.
        """
        input_dim = self.in_channels // self.groups
        conv_weight = torch.zeros(
            (self.in_channels, input_dim, self.kernel_size, self.kernel_size),
            dtype=branch_nrom.weight.dtype)

        for i in range(self.in_channels):
            conv_weight[i, i % input_dim, self.kernel_size // 2,
                        self.kernel_size // 2] = 1
        conv_weight = conv_weight.to(branch_nrom.weight.device)

        tmp_conv3x3 = self.create_conv_bn(kernel_size=3)
        tmp_conv3x3.conv.weight.data = conv_weight
        tmp_conv3x3.norm = branch_nrom
        return tmp_conv3x3


@BACKBONES.register_module()
class MobileOne(BaseBackbone):
    """MobileOne backbone.

    A PyTorch impl of : `An Improved One millisecond Mobile Backbone
    <https://arxiv.org/pdf/2206.04040.pdf>`_

    Args:
        arch (str | dict): The parameter of RepVGG.
            If it's a dict, it should contain the following keys:

            - num_blocks (Sequence[int]): Number of blocks in each stage.
            - width_factor (Sequence[float]): Width deflator in each stage.
            - group_layer_map (dict | None): RepVGG Block that declares
              the need to apply group convolution.
            - se_cfg (dict | None): Se Layer config

            Defaults to s0.

        in_channels (int): Number of input image channels. Default: 3.
        base_channels (int): Base channels of RepVGG backbone, work
            with width_factor together. Default: 64.
        out_indices (Sequence[int]): Output from which stages. Default: (3, ).
        strides (Sequence[int]): Strides of the first block of each stage.
            Default: (2, 2, 2, 2).
        dilations (Sequence[int]): Dilation of each stage.
            Default: (1, 1, 1, 1).
        frozen_stages (int): Stages to be frozen (all param fixed). -1 means
            not freezing any parameters. Default: -1.
        conv_cfg (dict | None): The config dict for conv layers. Default: None.
        norm_cfg (dict): The config dict for norm layers.
            Default: dict(type='BN').
        act_cfg (dict): Config dict for activation layer.
            Default: dict(type='ReLU').
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed. Default: False.
        deploy (bool): Whether to switch the model structure to deployment
            mode. Default: False.
        norm_eval (bool): Whether to set norm layers to eval mode, namely,
            freeze running stats (mean and var). Note: Effect on Batch Norm
            and its variants only. Default: False.
        init_cfg (dict or list[dict], optional): Initialization config dict.
    """

    arch_settings = {
        's0':
        dict(
            width_factor=[0.75, 1.0, 1.0, 2.0],
            num_conv_branches=[4, 4, 4, 4],
            num_se_blocks=[0, 0, 0, 0]),
        's1':
        dict(
            width_factor=[1.5, 1.5, 2.0, 2.5],
            num_conv_branches=[1, 1, 1, 1],
            num_se_blocks=[0, 0, 0, 0]),
        's2':
        dict(
            width_factor=[1.5, 2.0, 2.5, 4.0],
            num_conv_branches=[1, 1, 1, 1],
            num_se_blocks=[0, 0, 0, 0]),
        's3':
        dict(
            width_factor=[2.0, 2.5, 3.0, 4.0],
            num_conv_branches=[1, 1, 1, 1],
            num_se_blocks=[0, 0, 0, 0]),
        's4':
        dict(
            width_factor=[3.0, 3.5, 3.5, 4.0],
            num_conv_branches=[1, 1, 1, 1],
            num_se_blocks=[0, 0, 5, 1])
    }

    def __init__(self,
                 arch,
                 in_channels=3,
                 out_indices=(3, ),
                 num_blocks=[2, 8, 10, 1],
                 base_channels=[64, 128, 256, 512],
                 dilations=(1, 1, 1, 1),
                 frozen_stages=-1,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='ReLU'),
                 se_cfg=dict(ratio=16),
                 deploy=False,
                 norm_eval=False,
                 init_cfg=[
                     dict(type='Kaiming', layer=['Conv2d']),
                     dict(
                         type='Constant',
                         val=1,
                         layer=['_BatchNorm', 'GroupNorm'])
                 ]):
        super(MobileOne, self).__init__(init_cfg)

        if isinstance(arch, str):
            assert arch in self.arch_settings, f'"arch": "{arch}"' \
                f' is not one of the {list(self.arch_settings.keys())}'
            arch = self.arch_settings[arch]
        elif not isinstance(arch, dict):
            raise TypeError('Expect "arch" to be either a string '
                            f'or a dict, got {type(arch)}')

        self.arch = arch
        self.in_channels = in_channels
        self.base_channels = base_channels

        self.out_indices = out_indices
        self.deploy = deploy
        self.frozen_stages = frozen_stages
        self.norm_eval = norm_eval

        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.se_cfg = se_cfg
        self.act_cfg = act_cfg
        self.dilations = dilations

        channels = min(64,
                       int(base_channels[0] * self.arch['width_factor'][0]))
        self.stage0 = MobileOneBlock(
            self.in_channels,
            channels,
            stride=2,
            kernel_size=3,
            num_conv_branches=1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
            deploy=deploy)

        self.in_planes = channels
        self.stages = []
        for i, num_blocks in enumerate(num_blocks):
            dilation = self.dilations[i]
            planes = int(base_channels[i] * self.arch['width_factor'][i])

            stage = self._make_stage(planes, num_blocks, dilation,
                                     arch['num_se_blocks'][i],
                                     arch['num_conv_branches'][i])

            stage_name = f'stage{i + 1}'
            self.add_module(stage_name, stage)
            self.stages.append(stage_name)

    def _make_stage(self, planes, num_blocks, dilation, num_se,
                    num_conv_branches):
        strides = [2] + [1] * (num_blocks - 1)
        dilations = [dilation] * num_blocks
        if num_se > num_blocks:
            raise ValueError('Number of SE blocks cannot '
                             'exceed number of layers.')
        blocks = []
        for i in range(num_blocks):
            use_se = False
            if i >= (num_blocks - num_se):
                use_se = True

            blocks.append(
                # Depthwise conv
                MobileOneBlock(
                    in_channels=self.in_planes,
                    out_channels=self.in_planes,
                    kernel_size=3,
                    num_conv_branches=num_conv_branches,
                    stride=strides[i],
                    padding=1,
                    dilation=dilations[i],
                    groups=self.in_planes,
                    se_cfg=self.se_cfg if use_se else None,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg,
                    deploy=self.deploy))

            blocks.append(
                # Pointwise conv
                MobileOneBlock(
                    in_channels=self.in_planes,
                    out_channels=planes,
                    kernel_size=1,
                    num_conv_branches=num_conv_branches,
                    stride=1,
                    padding=0,
                    dilation=dilations[i],
                    se_cfg=self.se_cfg if use_se else None,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg,
                    deploy=self.deploy))

            self.in_planes = planes

        return Sequential(*blocks)

    def forward(self, x):
        x = self.stage0(x)
        outs = []
        for i, stage_name in enumerate(self.stages):
            stage = getattr(self, stage_name)
            x = stage(x)
            if i in self.out_indices:
                outs.append(x)

        return tuple(outs)

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            self.stem.eval()
            for param in self.stem.parameters():
                param.requires_grad = False
        for i in range(self.frozen_stages):
            stage = getattr(self, f'stage_{i+1}')
            stage.eval()
            for param in stage.parameters():
                param.requires_grad = False

    def train(self, mode=True):
        super(MobileOne, self).train(mode)
        self._freeze_stages()
        if mode and self.norm_eval:
            for m in self.modules():
                if isinstance(m, _BatchNorm):
                    m.eval()

    def switch_to_deploy(self):
        for m in self.modules():
            if isinstance(m, MobileOneBlock):
                m.switch_to_deploy()
        self.deploy = True
