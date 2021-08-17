# Copyright (c) OpenMMLab. All rights reserved.
from mmcv.utils import Registry, build_from_cfg

AUGMENT = Registry('augment')


def build_augment(cfg, default_args=None):
    return build_from_cfg(cfg, AUGMENT, default_args)
