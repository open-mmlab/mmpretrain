import torch.nn as nn
from mmcv.utils import Registry, build_from_cfg

BACKBONES = Registry('backbone')
CLASSIFIERS = Registry('classifier')
HEADS = Registry('head')
NECKS = Registry('neck')
LOSSES = Registry('loss')


def build(cfg, registry, default_args=None):
    if isinstance(cfg, list):
        modules = [
            build_from_cfg(cfg_, registry, default_args) for cfg_ in cfg
        ]
        return nn.Sequential(*modules)
    else:
        return build_from_cfg(cfg, registry, default_args)


def build_backbone(cfg):
    return build(cfg, BACKBONES)


def build_head(cfg):
    return build(cfg, HEADS)


def build_neck(cfg):
    return build(cfg, NECKS)


def build_loss(cfg):
    return build(cfg, LOSSES)


def build_classifier(cfg):
    return build(cfg, CLASSIFIERS)
