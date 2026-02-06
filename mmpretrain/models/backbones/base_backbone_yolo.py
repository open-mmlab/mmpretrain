from abc import ABCMeta, abstractmethod
from typing import List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn

from mmpretrain.registry import MODELS
from mmengine.model import BaseModule

@MODELS.register_module()
class BaseBackbone(BaseModule, metaclass=ABCMeta):
    """BaseBackbone backbone used in YOLO series.

    .. code:: text

     Backbone model structure diagram
     +-----------+
     |   input   |
     +-----------+
           v
     +-----------+
     |   stem    |
     |   layer   |
     +-----------+
           v
     +-----------+
     |   stage   |
     |  layer 1  |
     +-----------+
           v
     +-----------+
     |   stage   |
     |  layer 2  |
     +-----------+
           v
         ......
           v
     +-----------+
     |   stage   |
     |  layer n  |
     +-----------+
     In P5 model, n=4
     In P6 model, n=5

    Args:
        arch_setting (list): Architecture of BaseBackbone.
        plugins (list[dict]): List of plugins for stages, each dict contains:

            - cfg (dict, required): Cfg dict to build plugin.
            - stages (tuple[bool], optional): Stages to apply plugin, length
              should be same as 'num_stages'.
        deepen_factor (float): Depth multiplier, multiply number of
            blocks in CSP layer by this amount. Defaults to 1.0.
        widen_factor (float): Width multiplier, multiply number of
            channels in each layer by this amount. Defaults to 1.0.
        input_channels: Number of input image channels. Defaults to 3.
        out_indices (Sequence[int]): Output from which stages.
            Defaults to (2, 3, 4).
        frozen_stages (int): Stages to be frozen (stop grad and set eval
            mode). -1 means not freezing any parameters. Defaults to -1.
        norm_cfg (dict): Dictionary to construct and config norm layer.
            Defaults to None.
        act_cfg (dict): Config dict for activation layer.
            Defaults to None.
        norm_eval (bool): Whether to set norm layers to eval mode, namely,
            freeze running stats (mean and var). Note: Effect on Batch Norm
            and its variants only. Defaults to False.
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Defaults to None.
    """

    def __init__(self,
                 arch_setting: list,
                 deepen_factor: float = 1.0,
                 widen_factor: float = 1.0,
                 input_channels: int = 3,
                 out_indices: Sequence[int] = (2, 3, 4),
                 frozen_stages: int = -1,
                 plugins: Union[dict, List[dict]] = None,
                 norm_cfg = None,
                 act_cfg = None,
                 norm_eval: bool = False,
                 init_cfg = None):
        super().__init__(init_cfg)
        self.num_stages = len(arch_setting)
        self.arch_setting = arch_setting

        assert set(out_indices).issubset(
            i for i in range(len(arch_setting) + 1))

        if frozen_stages not in range(-1, len(arch_setting) + 1):
            raise ValueError('"frozen_stages" must be in range(-1, '
                             'len(arch_setting) + 1). But received '
                             f'{frozen_stages}')

        self.input_channels = input_channels
        self.out_indices = out_indices
        self.frozen_stages = frozen_stages
        self.widen_factor = widen_factor
        self.deepen_factor = deepen_factor
        self.norm_eval = norm_eval
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.plugins = plugins

        self.stem = self.build_stem_layer()
        self.layers = ['stem']

        for idx, setting in enumerate(arch_setting):
            stage = []
            stage += self.build_stage_layer(idx, setting)
            if plugins is not None:
                stage += self.make_stage_plugins(plugins, idx, setting)
            self.add_module(f'stage{idx + 1}', nn.Sequential(*stage))
            self.layers.append(f'stage{idx + 1}')

    @abstractmethod
    def build_stem_layer(self):
        """Build a stem layer."""
        pass

    @abstractmethod
    def build_stage_layer(self, stage_idx: int, setting: list):
        """Build a stage layer.

        Args:
            stage_idx (int): The index of a stage layer.
            setting (list): The architecture setting of a stage layer.
        """
        pass

    def make_stage_plugins(self, plugins, stage_idx, setting):
        """Make plugins for backbone ``stage_idx`` th stage.

        Currently we support to insert ``context_block``,
        ``empirical_attention_block``, ``nonlocal_block``, ``dropout_block``
        into the backbone.


        An example of plugins format could be:

        Examples:
            >>> plugins=[
            ...     dict(cfg=dict(type='xxx', arg1='xxx'),
            ...          stages=(False, True, True, True)),
            ...     dict(cfg=dict(type='yyy'),
            ...          stages=(True, True, True, True)),
            ... ]
            >>> model = YOLOv5CSPDarknet()
            >>> stage_plugins = model.make_stage_plugins(plugins, 0, setting)
            >>> assert len(stage_plugins) == 1

        Suppose ``stage_idx=0``, the structure of blocks in the stage would be:

        .. code-block:: none

            conv1 -> conv2 -> conv3 -> yyy

        Suppose ``stage_idx=1``, the structure of blocks in the stage would be:

        .. code-block:: none

            conv1 -> conv2 -> conv3 -> xxx -> yyy


        Args:
            plugins (list[dict]): List of plugins cfg to build. The postfix is
                required if multiple same type plugins are inserted.
            stage_idx (int): Index of stage to build
                If stages is missing, the plugin would be applied to all
                stages.
            setting (list): The architecture setting of a stage layer.

        Returns:
            list[nn.Module]: Plugins for current stage
        """
        # TODO: It is not general enough to support any channel and needs
        # to be refactored
        in_channels = int(setting[1] * self.widen_factor)
        plugin_layers = []
        for plugin in plugins:
            plugin = plugin.copy()
            stages = plugin.pop('stages', None)
            assert stages is None or len(stages) == self.num_stages
            if stages is None or stages[stage_idx]:
                name, layer = build_plugin_layer(
                    plugin['cfg'], in_channels=in_channels)
                plugin_layers.append(layer)
        return plugin_layers

    def _freeze_stages(self):
        """Freeze the parameters of the specified stage so that they are no
        longer updated."""
        if self.frozen_stages >= 0:
            for i in range(self.frozen_stages + 1):
                m = getattr(self, self.layers[i])
                m.eval()
                for param in m.parameters():
                    param.requires_grad = False

    def train(self, mode: bool = True):
        """Convert the model into training mode while keep normalization layer
        frozen."""
        super().train(mode)
        self._freeze_stages()
        if mode and self.norm_eval:
            for m in self.modules():
                if isinstance(m, _BatchNorm):
                    m.eval()

    def forward(self, x: torch.Tensor) -> tuple:
        """Forward batch_inputs from the data_preprocessor."""
        outs = []
        for i, layer_name in enumerate(self.layers):
            layer = getattr(self, layer_name)
            x = layer(x)
            if i in self.out_indices:
                outs.append(x)

        return tuple(outs)

