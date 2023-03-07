# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase

import torch
from torch import nn

from mmpretrain.engine import LearningRateDecayOptimWrapperConstructor
from mmpretrain.models import ImageClassifier, VisionTransformer


class ToyViTBackbone(nn.Module):

    get_layer_depth = VisionTransformer.get_layer_depth

    def __init__(self, num_layers=2):
        super().__init__()
        self.cls_token = nn.Parameter(torch.ones(1))
        self.pos_embed = nn.Parameter(torch.ones(1))
        self.num_layers = num_layers
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            layer = nn.Conv2d(3, 3, 1)
            self.layers.append(layer)


class ToyViT(nn.Module):
    get_layer_depth = ImageClassifier.get_layer_depth

    def __init__(self):
        super().__init__()
        # add some variables to meet unit test coverate rate
        self.backbone = ToyViTBackbone()
        self.head = nn.Linear(1, 1)


class TestLearningRateDecayOptimWrapperConstructor(TestCase):
    base_lr = 1.0
    base_wd = 0.05

    def test_add_params(self):
        model = ToyViT()
        optim_wrapper_cfg = dict(
            type='OptimWrapper',
            optimizer=dict(
                type='AdamW',
                lr=self.base_lr,
                betas=(0.9, 0.999),
                weight_decay=self.base_wd))
        paramwise_cfg = dict(
            layer_decay_rate=2.0,
            bias_decay_mult=0.,
            custom_keys={
                '.cls_token': dict(decay_mult=0.0),
                '.pos_embed': dict(decay_mult=0.0),
            })

        constructor = LearningRateDecayOptimWrapperConstructor(
            optim_wrapper_cfg=optim_wrapper_cfg,
            paramwise_cfg=paramwise_cfg,
        )
        optimizer_wrapper = constructor(model)

        expected_groups = [{
            'weight_decay': 0.0,
            'lr': 8 * self.base_lr,
            'param_name': 'backbone.cls_token',
        }, {
            'weight_decay': 0.0,
            'lr': 8 * self.base_lr,
            'param_name': 'backbone.pos_embed',
        }, {
            'weight_decay': self.base_wd,
            'lr': 4 * self.base_lr,
            'param_name': 'backbone.layers.0.weight',
        }, {
            'weight_decay': 0.0,
            'lr': 4 * self.base_lr,
            'param_name': 'backbone.layers.0.bias',
        }, {
            'weight_decay': self.base_wd,
            'lr': 2 * self.base_lr,
            'param_name': 'backbone.layers.1.weight',
        }, {
            'weight_decay': 0.0,
            'lr': 2 * self.base_lr,
            'param_name': 'backbone.layers.1.bias',
        }, {
            'weight_decay': self.base_wd,
            'lr': 1 * self.base_lr,
            'param_name': 'head.weight',
        }, {
            'weight_decay': 0.0,
            'lr': 1 * self.base_lr,
            'param_name': 'head.bias',
        }]
        self.assertIsInstance(optimizer_wrapper.optimizer, torch.optim.AdamW)
        self.assertEqual(optimizer_wrapper.optimizer.defaults['lr'],
                         self.base_lr)
        self.assertEqual(optimizer_wrapper.optimizer.defaults['weight_decay'],
                         self.base_wd)
        param_groups = optimizer_wrapper.optimizer.param_groups
        self.assertEqual(len(param_groups), len(expected_groups))

        for expect, param in zip(expected_groups, param_groups):
            self.assertEqual(param['param_name'], expect['param_name'])
            self.assertEqual(param['lr'], expect['lr'])
            self.assertEqual(param['weight_decay'], expect['weight_decay'])
