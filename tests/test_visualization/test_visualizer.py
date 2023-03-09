# Copyright (c) Open-MMLab. All rights reserved.
import os.path as osp
import tempfile
from unittest import TestCase
from unittest.mock import patch

import numpy as np
import torch

from mmpretrain.structures import DataSample
from mmpretrain.visualization import UniversalVisualizer


class TestUniversalVisualizer(TestCase):

    def setUp(self) -> None:
        super().setUp()
        tmpdir = tempfile.TemporaryDirectory()
        self.tmpdir = tmpdir
        self.vis = UniversalVisualizer(
            save_dir=tmpdir.name,
            vis_backends=[dict(type='LocalVisBackend')],
        )

    def test_visualize_cls(self):
        image = np.ones((10, 10, 3), np.uint8)
        data_sample = DataSample().set_gt_label(1).set_pred_label(1).\
            set_pred_score(torch.tensor([0.1, 0.8, 0.1]))

        # Test show
        def mock_show(drawn_img, win_name, wait_time):
            self.assertFalse((image == drawn_img).all())
            self.assertEqual(win_name, 'test_cls')
            self.assertEqual(wait_time, 0)

        with patch.object(self.vis, 'show', mock_show):
            self.vis.visualize_cls(
                image=image,
                data_sample=data_sample,
                show=True,
                name='test_cls',
                step=1)

        # Test storage backend.
        save_file = osp.join(self.tmpdir.name,
                             'vis_data/vis_image/test_cls_1.png')
        self.assertTrue(osp.exists(save_file))

        # Test out_file
        out_file = osp.join(self.tmpdir.name, 'results.png')
        self.vis.visualize_cls(
            image=image, data_sample=data_sample, out_file=out_file)
        self.assertTrue(osp.exists(out_file))

        # Test with dataset_meta
        self.vis.dataset_meta = {'classes': ['cat', 'bird', 'dog']}

        def patch_texts(text, *_, **__):
            self.assertEqual(
                text, '\n'.join([
                    'Ground truth: 1 (bird)',
                    'Prediction: 1, 0.80 (bird)',
                ]))

        with patch.object(self.vis, 'draw_texts', patch_texts):
            self.vis.visualize_cls(image, data_sample)

        # Test without pred_label
        def patch_texts(text, *_, **__):
            self.assertEqual(text, '\n'.join([
                'Ground truth: 1 (bird)',
            ]))

        with patch.object(self.vis, 'draw_texts', patch_texts):
            self.vis.visualize_cls(image, data_sample, draw_pred=False)

        # Test without gt_label
        def patch_texts(text, *_, **__):
            self.assertEqual(text, '\n'.join([
                'Prediction: 1, 0.80 (bird)',
            ]))

        with patch.object(self.vis, 'draw_texts', patch_texts):
            self.vis.visualize_cls(image, data_sample, draw_gt=False)

        # Test without score
        del data_sample.pred_score

        def patch_texts(text, *_, **__):
            self.assertEqual(
                text, '\n'.join([
                    'Ground truth: 1 (bird)',
                    'Prediction: 1 (bird)',
                ]))

        with patch.object(self.vis, 'draw_texts', patch_texts):
            self.vis.visualize_cls(image, data_sample)

        # Test adaptive font size
        def assert_font_size(target_size):

            def draw_texts(text, font_sizes, *_, **__):
                self.assertEqual(font_sizes, target_size)

            return draw_texts

        with patch.object(self.vis, 'draw_texts', assert_font_size(7)):
            self.vis.visualize_cls(
                np.ones((224, 384, 3), np.uint8), data_sample)

        with patch.object(self.vis, 'draw_texts', assert_font_size(2)):
            self.vis.visualize_cls(
                np.ones((10, 384, 3), np.uint8), data_sample)

        with patch.object(self.vis, 'draw_texts', assert_font_size(21)):
            self.vis.visualize_cls(
                np.ones((1000, 1000, 3), np.uint8), data_sample)

        # Test rescale image
        with patch.object(self.vis, 'draw_texts', assert_font_size(14)):
            self.vis.visualize_cls(
                np.ones((224, 384, 3), np.uint8),
                data_sample,
                rescale_factor=2.)

    def test_visualize_image_retrieval(self):
        image = np.ones((10, 10, 3), np.uint8)
        data_sample = DataSample().set_pred_score([0.1, 0.8, 0.1])

        class ToyPrototype:

            def get_data_info(self, idx):
                img_path = osp.join(osp.dirname(__file__), '../data/color.jpg')
                return {'img_path': img_path, 'sample_idx': idx}

        prototype_dataset = ToyPrototype()

        # Test show
        def mock_show(drawn_img, win_name, wait_time):
            if image.shape == drawn_img.shape:
                self.assertFalse((image == drawn_img).all())
            self.assertEqual(win_name, 'test_retrieval')
            self.assertEqual(wait_time, 0)

        with patch.object(self.vis, 'show', mock_show):
            self.vis.visualize_image_retrieval(
                image,
                data_sample,
                prototype_dataset,
                show=True,
                name='test_retrieval',
                step=1)

        # Test storage backend.
        save_file = osp.join(self.tmpdir.name,
                             'vis_data/vis_image/test_retrieval_1.png')
        self.assertTrue(osp.exists(save_file))

        # Test out_file
        out_file = osp.join(self.tmpdir.name, 'results.png')
        self.vis.visualize_image_retrieval(
            image,
            data_sample,
            prototype_dataset,
            out_file=out_file,
        )
        self.assertTrue(osp.exists(out_file))

    def test_visualize_masked_image(self):
        image = np.ones((10, 10, 3), np.uint8)
        data_sample = DataSample().set_mask(
            torch.tensor([
                [0, 0, 1, 1],
                [0, 1, 1, 0],
                [1, 1, 0, 0],
                [1, 0, 0, 1],
            ]))

        # Test show
        def mock_show(drawn_img, win_name, wait_time):
            self.assertTupleEqual(drawn_img.shape, (224, 224, 3))
            self.assertEqual(win_name, 'test_mask')
            self.assertEqual(wait_time, 0)

        with patch.object(self.vis, 'show', mock_show):
            self.vis.visualize_masked_image(
                image, data_sample, show=True, name='test_mask', step=1)

        # Test storage backend.
        save_file = osp.join(self.tmpdir.name,
                             'vis_data/vis_image/test_mask_1.png')
        self.assertTrue(osp.exists(save_file))

        # Test out_file
        out_file = osp.join(self.tmpdir.name, 'results.png')
        self.vis.visualize_masked_image(image, data_sample, out_file=out_file)
        self.assertTrue(osp.exists(out_file))

    def tearDown(self):
        self.tmpdir.cleanup()
