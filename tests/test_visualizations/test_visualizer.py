# Copyright (c) Open-MMLab. All rights reserved.
import os.path as osp
import tempfile
from unittest import TestCase
from unittest.mock import patch

import numpy as np
import torch

from mmpretrain.structures import ClsDataSample
from mmpretrain.visualization import ClsVisualizer


class TestClsVisualizer(TestCase):

    def setUp(self) -> None:
        super().setUp()
        tmpdir = tempfile.TemporaryDirectory()
        self.tmpdir = tmpdir
        self.vis = ClsVisualizer(
            save_dir=tmpdir.name,
            vis_backends=[dict(type='LocalVisBackend')],
        )

    def test_add_datasample(self):
        image = np.ones((10, 10, 3), np.uint8)
        data_sample = ClsDataSample().set_gt_label(1).set_pred_label(1).\
            set_pred_score(torch.tensor([0.1, 0.8, 0.1]))

        # Test show
        def mock_show(drawn_img, win_name, wait_time):
            self.assertFalse((image == drawn_img).all())
            self.assertEqual(win_name, 'test')
            self.assertEqual(wait_time, 0)

        with patch.object(self.vis, 'show', mock_show):
            self.vis.add_datasample(
                'test', image=image, data_sample=data_sample, show=True)

        # Test out_file
        out_file = osp.join(self.tmpdir.name, 'results.png')
        self.vis.add_datasample(
            'test', image=image, data_sample=data_sample, out_file=out_file)
        self.assertTrue(osp.exists(out_file))

        # Test storage backend.
        save_file = osp.join(self.tmpdir.name, 'vis_data/vis_image/test_0.png')
        self.assertTrue(osp.exists(save_file))

        # Test with dataset_meta
        self.vis.dataset_meta = {'classes': ['cat', 'bird', 'dog']}

        def test_texts(text, *_, **__):
            self.assertEqual(
                text, '\n'.join([
                    'Ground truth: 1 (bird)',
                    'Prediction: 1, 0.80 (bird)',
                ]))

        with patch.object(self.vis, 'draw_texts', test_texts):
            self.vis.add_datasample(
                'test', image=image, data_sample=data_sample)

        # Test without pred_label
        def test_texts(text, *_, **__):
            self.assertEqual(text, '\n'.join([
                'Ground truth: 1 (bird)',
            ]))

        with patch.object(self.vis, 'draw_texts', test_texts):
            self.vis.add_datasample(
                'test', image=image, data_sample=data_sample, draw_pred=False)

        # Test without gt_label
        def test_texts(text, *_, **__):
            self.assertEqual(text, '\n'.join([
                'Prediction: 1, 0.80 (bird)',
            ]))

        with patch.object(self.vis, 'draw_texts', test_texts):
            self.vis.add_datasample(
                'test', image=image, data_sample=data_sample, draw_gt=False)

        # Test without score
        del data_sample.pred_label.score

        def test_texts(text, *_, **__):
            self.assertEqual(
                text, '\n'.join([
                    'Ground truth: 1 (bird)',
                    'Prediction: 1 (bird)',
                ]))

        with patch.object(self.vis, 'draw_texts', test_texts):
            self.vis.add_datasample(
                'test', image=image, data_sample=data_sample)

        # Test adaptive font size
        def assert_font_size(target_size):

            def draw_texts(text, font_sizes, *_, **__):
                self.assertEqual(font_sizes, target_size)

            return draw_texts

        with patch.object(self.vis, 'draw_texts', assert_font_size(7)):
            self.vis.add_datasample(
                'test',
                image=np.ones((224, 384, 3), np.uint8),
                data_sample=data_sample)

        with patch.object(self.vis, 'draw_texts', assert_font_size(2)):
            self.vis.add_datasample(
                'test',
                image=np.ones((10, 384, 3), np.uint8),
                data_sample=data_sample)

        with patch.object(self.vis, 'draw_texts', assert_font_size(21)):
            self.vis.add_datasample(
                'test',
                image=np.ones((1000, 1000, 3), np.uint8),
                data_sample=data_sample)

        # Test rescale image
        with patch.object(self.vis, 'draw_texts', assert_font_size(14)):
            self.vis.add_datasample(
                'test',
                image=np.ones((224, 384, 3), np.uint8),
                rescale_factor=2.,
                data_sample=data_sample)

    def tearDown(self):
        self.tmpdir.cleanup()
