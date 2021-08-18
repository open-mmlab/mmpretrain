from mmcv.cnn import PLUGIN_LAYERS

from ..utils.se_layer import SELayer


@PLUGIN_LAYERS.register_module()
class SEPlugin(SELayer):

    def __init__(self, in_channels, **kwargs):
        super(SEPlugin, self).__init__(channels=in_channels, **kwargs)
