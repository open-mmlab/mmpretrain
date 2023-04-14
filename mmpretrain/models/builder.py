# Copyright (c) OpenMMLab. All rights reserved.
from mmpretrain.registry import MODELS

BACKBONES = MODELS
NECKS = MODELS
HEADS = MODELS
LOSSES = MODELS
CLASSIFIERS = MODELS
RETRIEVER = MODELS


def build_backbone(cfg):
    """Build backbone."""
    return BACKBONES.build(cfg)


def build_neck(cfg):
    """Build neck."""
    return NECKS.build(cfg)


def build_head(cfg):
    """Build head."""
    return HEADS.build(cfg)


def build_loss(cfg):
    """Build loss."""
    return LOSSES.build(cfg)


def build_classifier(cfg):
    """Build classifier."""
    return CLASSIFIERS.build(cfg)


def build_retriever(cfg):
    """Build retriever."""
    return RETRIEVER.build(cfg)
