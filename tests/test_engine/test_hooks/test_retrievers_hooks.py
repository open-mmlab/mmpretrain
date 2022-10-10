# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase
from unittest.mock import MagicMock

import torch

from mmcls.engine import ResetPrototypeInitFlagHook
from mmcls.models.retrievers import BaseRetriever


class ToyRetriever(BaseRetriever):
    def forward(self,
            inputs,
            data_samples = None,
            mode: str = 'loss'):
        pass

    def prepare_prototype(self):
        """Preprocessing the prototype before predict."""
        self.prototype_vecs = torch.tensor( [0] )
        self.prototype_inited = True


class TestClassNumCheckHook(TestCase):

    def setUp(self):
        self.hook = ResetPrototypeInitFlagHook()
        self.runner = MagicMock()
        self.runner.model = ToyRetriever()

    def test_before_val(self):
        self.runner.model.prepare_prototype()
        self.assertTrue(self.runner.model.prototype_inited) 
        self.hook.before_val(self.runner)
        self.assertFalse(self.runner.model.prototype_inited) 
