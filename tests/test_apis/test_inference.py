# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import ANY, MagicMock, patch

from mmcv.image import imread

from mmpretrain.apis import (ImageClassificationInferencer, ModelHub,
                             get_model, inference_model)
from mmpretrain.models import MobileNetV3
from mmpretrain.structures import DataSample
from mmpretrain.visualization import UniversalVisualizer

MODEL = 'mobilenet-v3-small-050_3rdparty_in1k'
WEIGHT = 'https://download.openmmlab.com/mmclassification/v0/mobilenet_v3/mobilenet-v3-small-050_3rdparty_in1k_20221114-e0b86be1.pth'  # noqa: E501
CONFIG = ModelHub.get(MODEL).config


class TestImageClassificationInferencer(TestCase):

    def test_init(self):
        # test input BaseModel
        model = get_model(MODEL)
        inferencer = ImageClassificationInferencer(model)
        self.assertEqual(model._config, inferencer.config)
        self.assertIsInstance(inferencer.model.backbone, MobileNetV3)

        # test input model name
        with patch('mmengine.runner.load_checkpoint') as mock:
            inferencer = ImageClassificationInferencer(MODEL)
            self.assertIsInstance(inferencer.model.backbone, MobileNetV3)
            mock.assert_called_once_with(ANY, WEIGHT, map_location='cpu')

        # test input config path
        inferencer = ImageClassificationInferencer(CONFIG.filename)
        self.assertIsInstance(inferencer.model.backbone, MobileNetV3)

        # test input config object
        inferencer = ImageClassificationInferencer(CONFIG)
        self.assertIsInstance(inferencer.model.backbone, MobileNetV3)

        # test specify weights
        with patch('mmengine.runner.load_checkpoint') as mock:
            ImageClassificationInferencer(MODEL, pretrained='custom.pth')
            mock.assert_called_once_with(ANY, 'custom.pth', map_location='cpu')

    def test_call(self):
        img_path = osp.join(osp.dirname(__file__), '../data/color.jpg')
        img = imread(img_path)

        # test inference classification model
        inferencer = ImageClassificationInferencer(MODEL)
        results = inferencer(img_path)[0]
        self.assertEqual(
            results.keys(),
            {'pred_score', 'pred_scores', 'pred_label', 'pred_class'})

        # test return_datasample=True
        results = inferencer(img, return_datasamples=True)[0]
        self.assertIsInstance(results, DataSample)

    def test_visualize(self):
        img_path = osp.join(osp.dirname(__file__), '../data/color.jpg')
        img = imread(img_path)

        inferencer = ImageClassificationInferencer(MODEL)
        self.assertIsNone(inferencer.visualizer)

        with TemporaryDirectory() as tmpdir:
            inferencer(img, show_dir=tmpdir)
            self.assertIsInstance(inferencer.visualizer, UniversalVisualizer)
            self.assertTrue(osp.exists(osp.join(tmpdir, '0.png')))

            inferencer.visualizer = MagicMock(wraps=inferencer.visualizer)
            inferencer(
                img_path, rescale_factor=2., draw_score=False, show_dir=tmpdir)
            self.assertTrue(osp.exists(osp.join(tmpdir, 'color.png')))
            inferencer.visualizer.visualize_cls.assert_called_once_with(
                ANY,
                ANY,
                classes=inferencer.classes,
                resize=None,
                show=False,
                wait_time=0,
                rescale_factor=2.,
                draw_gt=False,
                draw_pred=True,
                draw_score=False,
                name='color',
                out_file=osp.join(tmpdir, 'color.png'))


class TestInferenceAPIs(TestCase):

    def test_inference_model(self):
        # test backward compatibility
        img_path = osp.join(osp.dirname(__file__), '../data/color.jpg')
        img = imread(img_path)

        model = get_model(MODEL, pretrained=True)
        results = inference_model(model, img_path)
        self.assertEqual(
            results.keys(),
            {'pred_score', 'pred_scores', 'pred_label', 'pred_class'})

        results = inference_model(model, img)
        self.assertEqual(
            results.keys(),
            {'pred_score', 'pred_scores', 'pred_label', 'pred_class'})

        # test input model name
        results = inference_model(MODEL, img)
        self.assertEqual(
            results.keys(),
            {'pred_score', 'pred_scores', 'pred_label', 'pred_class'})
