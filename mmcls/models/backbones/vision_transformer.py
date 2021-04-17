# import torch
# import torch.nn as nn
# from mmcv.cnn import build_activation_layer, build_norm_layer, xavier_init
# from mmcv.cnn.bricks.registry import (TRANSFORMER_LAYER,
#                                       TRANSFORMER_LAYER_SEQUENCE)
# from mmcv.cnn.bricks.transformer import (BaseTransformerLayer,
#                                          TransformerLayerSequence,
#                                          build_transformer_layer_sequence)
# from mmcv.runner.base_module import BaseModule
#
#
# @TRANSFORMER_LAYER.register_module()
# class TransformerEncoderLayer(BaseTransformerLayer):
#     """Implements encoder layer in DETR transformer.
#
#     Args:
#         attn_cfgs (list[`mmcv.ConfigDict`] | list[dict] | dict )):
#             Configs for self_attention or cross_attention, the order
#             should be consistent with it in `operation_order`. If it is
#             a dict, it would be expand to the number of attention in
#             `operation_order`.
#         feedforward_channels (int): The hidden dimension for FFNs.
#         ffn_dropout (float): Probability of an element to be zeroed
#             in ffn. Default 0.0.
#         operation_order (tuple[str]): The execution order of operation
#             in transformer. Such as ('self_attn', 'norm', 'ffn', 'norm').
#             Default：None
#         act_cfg (dict): The activation config for FFNs.
#         norm_cfg (dict): Config dict for normalization layer.
#         ffn_num_fcs (int): The number of fully-connected layers in FFNs.
#             Default：2.
#     """
#
#     def __init__(self,
#                  attn_cfgs,
#                  feedforward_channels,
#                  ffn_dropout=0.0,
#                  operation_order=None,
#                  act_cfg=dict(type='ReLU', inplace=True),
#                  norm_cfg=dict(type='LN'),
#                  ffn_num_fcs=2,
#                  **kwargs):
#         super(TransformerEncoderLayer, self).__init__(
#             attn_cfgs=attn_cfgs,
#             feedforward_channels=feedforward_channels,
#             operation_order=operation_order,
#             ffn_dropout=ffn_dropout,
#             act_cfg=act_cfg,
#             norm_cfg=norm_cfg,
#             ffn_num_fcs=ffn_num_fcs,
#             **kwargs)
#         assert len(self.operation_order) == 4
#         assert set(self.operation_order) == set(['self_attn', 'norm', 'ffn'])
