import torch
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from mmcv.cnn import build_activation_layer, build_conv_layer, build_norm_layer
from mmcv.runner import BaseModule
from mmcv.utils.parrots_wrapper import _BatchNorm
from torch import nn

from ..builder import BACKBONES
from ..utils.se_layer import SELayer
from .base_backbone import BaseBackbone


class RepVGGBlock(BaseModule):
    """RepVGG block for RepVGG backbone.

    Args:
        in_channels (int): The input channels of the block.
        out_channels (int): The output channels of the block.
        stride (int): Stride of the 3x3 and 1x1 convolution layer. Default: 1.
        padding (int): Padding of the 3x3 convolution layer.
        dilation (int): Dilation of the 3x3 convolution layer.
        groups (int): Groups of the 3x3 and 1x1 convolution layer. Default: 1.
        padding_mode (str): Padding mode of the 3x3 convolution layer.
            Default: 'zeros'.
        se_cfg (None or dict): The configuration of the se module.
            Default: None.
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed. Default: False.
        conv_cfg (dict, optional): Config dict for convolution layer.
            Default: None, which means using conv2d.
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
                 stride=1,
                 padding=1,
                 dilation=1,
                 groups=1,
                 padding_mode='zeros',
                 se_cfg=None,
                 with_cp=False,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='ReLU'),
                 deploy=False,
                 init_cfg=None):
        super(RepVGGBlock, self).__init__(init_cfg)

        assert se_cfg is None or isinstance(se_cfg, dict)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.se_cfg = se_cfg
        self.with_cp = with_cp
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.deploy = deploy

        if deploy:
            self.branch_reparam = build_conv_layer(
                conv_cfg,
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
                bias=True,
                padding_mode=padding_mode)
        else:
            self.branch_norm = build_norm_layer(norm_cfg, in_channels)[1] \
                if out_channels == in_channels and stride == 1 and \
                padding == dilation else None
            self.branch_3x3 = self.create_conv_bn(
                kernel_size=3,
                dilation=dilation,
                padding=padding,
            )
            self.branch_1x1 = self.create_conv_bn(kernel_size=1)

        if se_cfg is not None:
            if 'init_cfg' in se_cfg.keys():
                self.se_layer = SELayer(channels=out_channels, **se_cfg)
            else:
                self.se_layer = SELayer(
                    channels=out_channels, init_cfg=init_cfg, **se_cfg)

        self.act = build_activation_layer(act_cfg)

    def create_conv_bn(self, kernel_size, dilation=1, padding=0):
        result = nn.Sequential()
        result.add_module(
            'conv',
            build_conv_layer(
                self.conv_cfg,
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                kernel_size=kernel_size,
                stride=self.stride,
                dilation=dilation,
                padding=padding,
                groups=self.groups,
                bias=False))
        result.add_module(
            'norm',
            build_norm_layer(self.norm_cfg, num_features=self.out_channels)[1])

        return result

    def forward(self, x):

        def _inner_forward(inputs):
            if self.deploy:
                return self.branch_reparam(inputs)

            if self.branch_norm is None:
                branch_norm_out = 0
            else:
                branch_norm_out = self.branch_norm(inputs)

            inner_out = self.branch_3x3(inputs) + self.branch_1x1(
                inputs) + branch_norm_out

            if self.se_cfg is not None:
                inner_out = self.se_layer(inner_out)

            return inner_out

        if self.with_cp and x.requires_grad:
            out = cp.checkpoint(_inner_forward, x)
        else:
            out = _inner_forward(x)

        out = self.act(out)

        return out

    def switch_to_deploy(self):
        """Switch the model structure from training mode to deployment mode.

        Returns:
            None
        """
        if self.deploy:
            return
        assert self.norm_cfg['type'] == 'BN', \
            "Switch is not allowed when norm_cfg['type'] != 'BN'."

        reparam_weight, reparam_bias = self.reparameterize()
        self.branch_reparam = build_conv_layer(
            self.conv_cfg,
            self.branch_3x3.conv.in_channels,
            self.branch_3x3.conv.out_channels,
            kernel_size=3,
            stride=self.branch_3x3.conv.stride,
            padding=self.branch_3x3.conv.padding,
            dilation=self.branch_3x3.conv.dilation,
            groups=self.branch_3x3.conv.groups,
            bias=True)
        self.branch_reparam.weight.data = reparam_weight
        self.branch_reparam.bias.data = reparam_bias

        for param in self.parameters():
            param.detach_()
        self.__delattr__('branch_3x3')
        self.__delattr__('branch_1x1')
        self.__delattr__('branch_identity')

    def reparameterize(self):
        """Fuse all the parameters of all branchs.

        Returns:
            (Weight, bias) (tuple): Parameters after fusion of all branches.
                the first element is the weight and the second is the bias.
        """
        weight_3x3, bias_3x3 = self._fuse_one_branch(self.branch_3x3)
        weight_1x1, bias_1x1 = self._fuse_one_branch(self.branch_1x1)
        weight_1x1 = F.pad(weight_1x1, [1, 1, 1, 1])
        weight_identity, bias_identity = self._fuse_one_branch(
            self.branch_norm)

        return (weight_3x3 + weight_1x1 + weight_identity,
                bias_3x3 + bias_1x1 + bias_identity)

    def _fuse_one_branch(self, branch):
        """Fuse the parameters of conv and bn in one branch of repvgg.

        Args:
            branch (None, nn.Sequential or nn.BatchNorm2d): A branch
                in repvggblock.

        Returns:
            (weight, bias) (tuple): The parameters obtained after
                fusing the parameters of conv and bn in one branch.
                The first element is the weight and the second is the bias.
        """
        if branch is None:
            return 0, 0

        if isinstance(branch, nn.Sequential):
            return self._fuse_conv_bn(branch)
        else:
            return self._fuse_norm(branch)

    @staticmethod
    def _fuse_conv_bn(self, branch):
        """Fuse the parameters in a branch with a conv and bn.

        Args:
            branch (None, nn.Sequential or nn.BatchNorm2d): A branch
                with conv and bn in repvggblock.

        Returns:
            (weight, bias) (tuple): The parameters obtained after
                fusing the parameters of conv and bn in one branch.
                The first element is the weight and the second is the bias.
        """
        conv_weight = branch.conv.weight
        running_mean = branch.norm.running_mean
        running_var = branch.norm.running_var
        gamma = branch.norm.weight
        beta = branch.norm.bias
        eps = branch.norm.eps

        std = (running_var + eps).sqrt()
        fused_weight = (gamma / std).reshape(-1, 1, 1, 1) * conv_weight
        fused_bias = -running_mean * gamma / std + beta

        return fused_weight, fused_bias

    def _fuse_norm(self, branch):
        """Fuse the parameters in a branch with a conv and bn.

        Args:
            branch (None, nn.Sequential or nn.BatchNorm2d): A branch
                only with bn in repvggblock.

        Returns:
            (weight, bias) (tuple): The parameters obtained after
                fusing the parameters of conv and bn in one branch.
                The first element is the weight and the second is the bias.
        """
        input_dim = self.in_channels // self.groups
        conv_weight = torch.zeros((self.in_channels, input_dim, 3, 3),
                                  dtype=branch.weight.dtype)
        for i in range(self.in_channels):
            conv_weight[i, i % input_dim, 1, 1] = 1
        conv_weight = conv_weight.to(branch.weight.device)
        running_mean = branch.running_mean
        running_var = branch.running_var
        gamma = branch.weight
        beta = branch.bias
        eps = branch.eps

        std = (running_var + eps).sqrt()
        fused_weight = (gamma / std).reshape(-1, 1, 1, 1) * conv_weight
        fused_bias = -running_mean * gamma / std + beta

        return fused_weight, fused_bias


@BACKBONES.register_module()
class RepVGG(BaseBackbone):
    """RepVGG backbone.

    Args:
        arch (str | dict): The parameter of RepVGG
            - num_blocks (Sequence[int]): Number of blocks in each stage.
            - width_factor (Sequence[float]): Width deflator in each stage.
            - group_idx (dict | optional): RepVGG Block that declares
                the need to apply group convolution.
        in_channels (int): Number of input image channels. Default: 3.
        base_channels (int): Base channels of RepVGG backbone, work
            with width_factor together.
        out_indices (Sequence[int]): Output from which stages.
        strides (Sequence[int]): Strides of the first block of each stage.
        dilations (Sequence[int]): Dilation of each stage.
        se_cfg (None or dict): The configuration of the se module.
            Default: None.
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
        frozen_stages (int): Stages to be frozen (all param fixed). -1 means
            not freezing any parameters. Default: -1.
        conv_cfg (dict | None): The config dict for conv layers. Default: None.
        norm_cfg (dict): The config dict for norm layers.
        act_cfg (dict): Config dict for activation layer.
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed. Default: False.
        deploy (bool): Whether to switch the model structure to deployment
            mode. Default: False.
        norm_eval (bool): Whether to set norm layers to eval mode, namely,
            freeze running stats (mean and var). Note: Effect on Batch Norm
            and its variants only. Default: False.
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None
    """

    optional_groupwise_layers = [
        2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26
    ]
    g2_map = {layer: 2 for layer in optional_groupwise_layers}
    g4_map = {layer: 4 for layer in optional_groupwise_layers}

    arch_settings = {
        'A0':
        dict(num_blocks=[2, 4, 14, 1], width_factor=[0.75, 0.75, 0.75, 2.5]),
        'A1':
        dict(num_blocks=[2, 4, 14, 1], width_factor=[1, 1, 1, 2.5]),
        'A2':
        dict(num_blocks=[2, 4, 14, 1], width_factor=[1.5, 1.5, 1.5, 2.75]),
        'B0':
        dict(num_blocks=[4, 6, 16, 1], width_factor=[1, 1, 1, 2.5]),
        'B1':
        dict(num_blocks=[4, 6, 16, 1], width_factor=[2, 2, 2, 4]),
        'B1g2':
        dict(
            num_blocks=[4, 6, 16, 1],
            width_factor=[2, 2, 2, 4],
            group_idx=g2_map),
        'B1g4':
        dict(
            num_blocks=[4, 6, 16, 1],
            width_factor=[2, 2, 2, 4],
            group_idx=g4_map),
        'B2':
        dict(num_blocks=[4, 6, 16, 1], width_factor=[2.5, 2.5, 2.5, 5]),
        'B2g2':
        dict(
            num_blocks=[4, 6, 16, 1],
            width_factor=[2.5, 2.5, 2.5, 5],
            group_idx=g2_map),
        'B2g4':
        dict(
            num_blocks=[4, 6, 16, 1],
            width_factor=[2.5, 2.5, 2.5, 5],
            group_idx=g4_map),
        'B3':
        dict(num_blocks=[4, 6, 16, 1], width_factor=[3, 3, 3, 5]),
        'B3g2':
        dict(
            num_blocks=[4, 6, 16, 1],
            width_factor=[3, 3, 3, 5],
            group_idx=g2_map),
        'B3g4':
        dict(
            num_blocks=[4, 6, 16, 1],
            width_factor=[3, 3, 3, 5],
            group_idx=g4_map),
        'D2se':
        dict(num_blocks=[8, 14, 24, 1], width_factor=[2.5, 2.5, 2.5, 5])
    }

    def __init__(self,
                 arch,
                 in_channels=3,
                 base_channels=64,
                 out_indices=(3, ),
                 strides=(2, 2, 2, 2),
                 dilations=(1, 1, 1, 1),
                 se_cfg=None,
                 frozen_stages=-1,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='ReLU'),
                 with_cp=False,
                 deploy=False,
                 norm_eval=False,
                 init_cfg=[
                     dict(type='Kaiming', layer=['Conv2d']),
                     dict(
                         type='Constant',
                         val=1,
                         layer=['_BatchNorm', 'GroupNorm'])
                 ]):
        super(RepVGG, self).__init__(init_cfg)

        if isinstance(arch, str):
            assert arch in self.arch_settings, \
                f'"arch": "{arch}" is not one of the arch_settings'
            arch = self.arch_settings[arch]
        elif not isinstance(arch, dict):
            raise TypeError('Expect "arch" to be either a string '
                            f'or a dict, got {type(arch)}')

        assert len(arch['num_blocks']) == len(
            arch['width_factor']) == len(strides) == len(dilations)
        assert max(out_indices) < len(arch['num_blocks'])
        if 'group_idx' in arch.keys():
            assert max(arch['group_idx'].keys()) <= sum(arch['num_blocks'])

        self.arch = arch
        self.in_channels = in_channels
        self.base_channels = base_channels
        self.out_indices = out_indices
        self.strides = strides
        self.dilations = dilations
        self.se_cfg = se_cfg
        self.deploy = deploy
        self.frozen_stages = frozen_stages
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.with_cp = with_cp
        self.norm_eval = norm_eval

        channels = min(64, int(base_channels * self.arch['width_factor'][0]))
        self.stage_0 = RepVGGBlock(
            self.in_channels,
            channels,
            stride=2,
            se_cfg=se_cfg,
            with_cp=with_cp,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
            deploy=deploy)

        next_create_block_idx = 1
        self.stages = []
        for i in range(len(arch['num_blocks'])):
            num_blocks = self.arch['num_blocks'][i]
            stride = self.strides[i]
            dilation = self.dilations[i]
            out_channels = int(base_channels * 2**i *
                               self.arch['width_factor'][i])

            stage, next_create_block_idx = self._make_stage(
                channels, out_channels, num_blocks, stride, dilation,
                next_create_block_idx, se_cfg, init_cfg)
            stage_name = f'stage_{i + 1}'
            self.add_module(stage_name, stage)
            self.stages.append(stage_name)

            channels = out_channels

    def _make_stage(self, in_channels, out_channels, num_blocks, stride,
                    dilation, next_create_block_idx, se_cfg, init_cfg):
        strides = [stride] + [1] * (num_blocks - 1)
        dilations = [dilation] * num_blocks

        blocks = []
        for i in range(num_blocks):
            groups = self.arch['group_idx'].get(
                next_create_block_idx,
                1) if 'group_idx' in self.arch.keys() else 1
            blocks.append(
                RepVGGBlock(
                    in_channels,
                    out_channels,
                    stride=strides[i],
                    padding=dilations[i],
                    dilation=dilations[i],
                    groups=groups,
                    se_cfg=se_cfg,
                    with_cp=self.with_cp,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg,
                    deploy=self.deploy,
                    init_cfg=init_cfg))
            in_channels = out_channels
            next_create_block_idx += 1

        return nn.Sequential(*blocks), next_create_block_idx

    def forward(self, x):
        x = self.stage_0(x)
        outs = []
        for i, stage_name in enumerate(self.stages):
            stage = getattr(self, stage_name)
            x = stage(x)
            if i in self.out_indices:
                outs.append(x)

        if len(outs) == 1:
            return outs[0]
        else:
            return tuple(outs)

    def _freeze_stages(self):
        for i in range(self.frozen_stages + 1):
            stage = getattr(self, f'stage_{i}')
            stage.eval()
            for param in stage.parameters():
                param.requires_grad = False

    def train(self, mode=True):
        super(RepVGG, self).train(mode)
        self._freeze_stages()
        if mode and self.norm_eval:
            for m in self.modules():
                if isinstance(m, _BatchNorm):
                    m.eval()
