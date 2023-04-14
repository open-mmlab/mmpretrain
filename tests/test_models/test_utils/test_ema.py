# Copyright (c) OpenMMLab. All rights reserved.
import math
from unittest import TestCase

import torch
import torch.nn as nn
from mmengine.logging import MessageHub
from mmengine.testing import assert_allclose

from mmpretrain.models.utils import CosineEMA


class TestEMA(TestCase):

    def test_cosine_ema(self):
        model = nn.Sequential(nn.Conv2d(1, 5, kernel_size=3), nn.Linear(5, 10))

        # init message hub
        max_iters = 5
        test = dict(name='ema_test')
        message_hub = MessageHub.get_instance(**test)
        message_hub.update_info('max_iters', max_iters)

        # test EMA
        momentum = 0.996
        end_momentum = 1.

        ema_model = CosineEMA(model, momentum=1 - momentum)
        averaged_params = [
            torch.zeros_like(param) for param in model.parameters()
        ]

        for i in range(max_iters):
            updated_averaged_params = []
            for p, p_avg in zip(model.parameters(), averaged_params):
                p.detach().add_(torch.randn_like(p))
                if i == 0:
                    updated_averaged_params.append(p.clone())
                else:
                    m = end_momentum - (end_momentum - momentum) * (
                        math.cos(math.pi * i / float(max_iters)) + 1) / 2
                    updated_averaged_params.append(
                        (p_avg * m + p * (1 - m)).clone())
            ema_model.update_parameters(model)
            averaged_params = updated_averaged_params

        for p_target, p_ema in zip(averaged_params, ema_model.parameters()):
            assert_allclose(p_target, p_ema)
