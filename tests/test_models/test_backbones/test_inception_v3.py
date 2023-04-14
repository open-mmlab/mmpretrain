# Copyright (c) OpenMMLab. All rights reserved.
from types import MethodType
from unittest import TestCase

import torch

from mmpretrain.models import InceptionV3
from mmpretrain.models.backbones.inception_v3 import InceptionAux


class TestInceptionV3(TestCase):
    DEFAULT_ARGS = dict(num_classes=10, aux_logits=False, dropout=0.)

    def test_structure(self):
        # Test without auxiliary branch.
        model = InceptionV3(**self.DEFAULT_ARGS)
        self.assertIsNone(model.AuxLogits)

        # Test with auxiliary branch.
        cfg = {**self.DEFAULT_ARGS, 'aux_logits': True}
        model = InceptionV3(**cfg)
        self.assertIsInstance(model.AuxLogits, InceptionAux)

    def test_init_weights(self):
        cfg = {**self.DEFAULT_ARGS, 'aux_logits': True}
        model = InceptionV3(**cfg)

        init_info = {}

        def get_init_info(self, *args):
            for name, param in self.named_parameters():
                init_info[name] = ''.join(
                    self._params_init_info[param]['init_info'])

        model._dump_init_info = MethodType(get_init_info, model)
        model.init_weights()
        self.assertIn('TruncNormalInit: a=-2, b=2, mean=0, std=0.1, bias=0',
                      init_info['Conv2d_1a_3x3.conv.weight'])
        self.assertIn('TruncNormalInit: a=-2, b=2, mean=0, std=0.01, bias=0',
                      init_info['AuxLogits.conv0.conv.weight'])
        self.assertIn('TruncNormalInit: a=-2, b=2, mean=0, std=0.001, bias=0',
                      init_info['AuxLogits.fc.weight'])

    def test_forward(self):
        inputs = torch.rand(2, 3, 299, 299)

        model = InceptionV3(**self.DEFAULT_ARGS)
        aux_out, out = model(inputs)
        self.assertIsNone(aux_out)
        self.assertEqual(out.shape, (2, 10))

        cfg = {**self.DEFAULT_ARGS, 'aux_logits': True}
        model = InceptionV3(**cfg)
        aux_out, out = model(inputs)
        self.assertEqual(aux_out.shape, (2, 10))
        self.assertEqual(out.shape, (2, 10))
