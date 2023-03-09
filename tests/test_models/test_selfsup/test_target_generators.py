# Copyright (c) OpenMMLab. All rights reserved.
import platform
from unittest import TestCase

import pytest
import torch

from mmpretrain.models import VQKD, DALLEEncoder, HOGGenerator


class TestDALLE(TestCase):

    @pytest.mark.skipif(
        platform.system() == 'Windows', reason='Windows mem limit')
    def test_dalle(self):
        model = DALLEEncoder()
        fake_inputs = torch.rand((2, 3, 112, 112))
        fake_outputs = model(fake_inputs)

        assert list(fake_outputs.shape) == [2, 8192, 14, 14]


class TestHOGGenerator(TestCase):

    def test_hog_generator(self):
        hog_generator = HOGGenerator()

        fake_input = torch.randn((2, 3, 224, 224))
        fake_output = hog_generator(fake_input)
        assert list(fake_output.shape) == [2, 196, 108]

        fake_hog_out = hog_generator.out[0].unsqueeze(0)
        fake_hog_img = hog_generator.generate_hog_image(fake_hog_out)
        assert fake_hog_img.shape == (224, 224)

        with pytest.raises(AssertionError):
            fake_hog_img = hog_generator.generate_hog_image(
                hog_generator.out[0])


class TestVQKD(TestCase):

    ENCODER_CFG = dict(
        arch='base',
        img_size=224,
        patch_size=16,
        in_channels=3,
        out_indices=-1,
        drop_rate=0.,
        drop_path_rate=0.,
        norm_cfg=dict(type='LN', eps=1e-6),
        final_norm=True,
        out_type='featmap',
        with_cls_token=True,
        frozen_stages=-1,
        use_abs_pos_emb=True,
        use_rel_pos_bias=False,
        use_shared_rel_pos_bias=False,
        layer_scale_init_value=0.,
        interpolate_mode='bicubic',
        patch_cfg=dict(),
        layer_cfgs=dict(),
        init_cfg=None)

    @pytest.mark.skipif(
        platform.system() == 'Windows', reason='Windows mem limit')
    def test_vqkd(self):
        model = VQKD(encoder_config=self.ENCODER_CFG)
        fake_inputs = torch.rand((2, 3, 224, 224))
        fake_outputs = model(fake_inputs)

        assert list(fake_outputs.shape) == [2, 196]
