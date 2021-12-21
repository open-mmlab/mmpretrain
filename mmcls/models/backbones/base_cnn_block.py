# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Dict, Any, Optional

from mmcv.cnn import build_activation_layer, build_plugin_layer
from abc import ABCMeta, abstractmethod
from mmcv.runner import BaseModule

PLUGIN_DICT=Dict[str, Optional[Any]]

class PluginsBaseCNNBlock(BaseModule, metaclass=ABCMeta):
    """Baseblock for CNN Neural Networks backbone with Plugins;

    Args:
        in_channels (int): The input channels of the block.
        out_channels (int): The output channels of the block.
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None
    """

    def __init__(self,
                 plugins_allowed_position:List[str]=[],
                 plugins:Optional[List[PLUGIN_DICT]]=None,
                 init_cfg=None):
        super(PluginsBaseCNNBlock, self).__init__(init_cfg)

        self.with_plugins = plugins is not None
        self.allowed_position = plugins_allowed_position
        self.plugins = plugins

        assert plugins is None or isinstance(plugins, list)
        if self.with_plugins:
            assert all(p['position'] in self.allowed_position for p in plugins), \
                f"'position' must be in {','.join(self.allowed_position)}."

        self.plugin_dict = {p:list() for p in self.allowed_position}
        self.position2inchannels_map = dict()

    def make_block_plugins(self, plugins):
        """make plugins for block.
        Args:
            plugins (list[dict]): List of plugins cfg to build.
        Returns:
            list[str]: List of the names of plugin.
        """
        assert isinstance(plugins, list)
        plugin_names = []
        for plugin in plugins:
            position_name = plugin['position']
            plugin = plugin.copy()['cfg']
            in_channels = self.position2inchannels_map[position_name]
            name, layer = build_plugin_layer(
                plugin,
                channels=in_channels,
                postfix=plugin.pop('postfix', ''))
            assert not hasattr(self, name), f'duplicate plugin {name}'
            self.add_module(name, layer)
            plugin_names.append(name)
        return plugin_names

    @abstractmethod
    def get_position2inchannels_map(self):
        """get the input channels in different place"""
        raise NotImplementedError


    def build_plugins(self):
        if not self.with_plugins:
            return 
        self.position2inchannels_map = self.get_position2inchannels_map()
        assert set(self.position2inchannels_map.keys()) == set(self.allowed_position)

        for position_name in self.allowed_position:
            position_plugin_cfgs = [
                plugin for plugin in self.plugins
                if plugin['position'] == position_name
            ]

            plugin_layer_names = self.make_block_plugins(position_plugin_cfgs)
            self.plugin_dict[position_name] = plugin_layer_names
        
    def forward_plugin(self, x, position):
        """Forward function for plugins."""
        out = x
        plugin_names = self.plugin_dict[position]
        for name in plugin_names:
            out = getattr(self, name)(x)
        return out


class BaseBottleNeck(PluginsBaseCNNBlock):
    """BaseBottleNeck for ResNet backbone with Plugins;

    Args:
        in_channels (int): The input channels of the block.
        out_channels (int): The output channels of the block.
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None
    """
    def __init__(self,
            in_channels,
            out_channels,
            plugins=None,
            init_cfg=None):
        super(BaseBottleNeck, self).__init__(
            plugins_allowed_position=[f"after_conv{i}" for i in range(1, 4)], 
            plugins=plugins,
            init_cfg=init_cfg)

        self.in_channels = in_channels
        self.out_channels = out_channels

    @abstractmethod
    def get_mid_channels(self):
        raise NotImplementedError

    def get_position2inchannels_map(self):
        """get the input channels in different place"""
        return dict(
            after_conv1=self.mid_channels,
            after_conv2=self.mid_channels,
            after_conv3=self.out_channels)  


class BaseBasicBlock(PluginsBaseCNNBlock):
    """Baseblock for CNN Neural Networks backbone with Plugins;

    Args:
        in_channels (int): The input channels of the block.
        out_channels (int): The output channels of the block.
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None
    """
    def __init__(self,
                in_channels,
                out_channels,
                plugins=None,
                init_cfg=None):
        super(BaseBasicBlock, self).__init__(
            plugins_allowed_position=["after_conv1", "after_conv2"], 
            plugins=plugins, 
            init_cfg=init_cfg)

        self.in_channels = in_channels
        self.out_channels = out_channels

    @abstractmethod
    def get_mid_channels(self):
        raise NotImplementedError

    def get_position2inchannels_map(self):
        """get the input channels in different place"""
        return dict(
            after_conv1=self.mid_channels,
            after_conv2=self.out_channels)


class BaseVGGBlock(PluginsBaseCNNBlock):
    """Baseblock for CNN Neural Networks backbone with Plugins;

    Args:
        in_channels (int): The input channels of the block.
        out_channels (int): The output channels of the block.
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None
    """
    def __init__(self,
                in_channels,
                out_channels,
                plugins=None,
                init_cfg=None):
        super(BaseVGGBlock, self).__init__(
            plugins_allowed_position=["after_conv1"],
            plugins=plugins, 
            init_cfg=init_cfg)
        self.in_channels = in_channels
        self.out_channels = out_channels

    def get_position2inchannels_map(self):
        """get the input channels in different place"""
        return dict(after_conv1=self.out_channels)
