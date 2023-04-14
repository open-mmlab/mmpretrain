# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
from unittest import TestCase
from unittest.mock import patch

from mmengine import Config

from mmpretrain.apis import ModelHub, get_model, init_model, list_models
from mmpretrain.models import ImageClassifier, MobileNetV2


class TestModelHub(TestCase):

    def test_mmpretrain_models(self):
        self.assertIn('resnet18_8xb32_in1k', ModelHub._models_dict)

    def test_register_model_index(self):
        model_index_path = osp.join(osp.dirname(__file__), '../data/meta.yml')

        ModelHub.register_model_index(model_index_path)
        self.assertIn('test_model', ModelHub._models_dict)
        self.assertEqual(
            ModelHub._models_dict['test_model'].config,
            osp.abspath(
                osp.join(osp.dirname(model_index_path), 'test_config.py')))

        with self.assertRaisesRegex(ValueError, 'meta.yml'):
            # test name conflict
            ModelHub.register_model_index(model_index_path)

        # test specify config prefix
        del ModelHub._models_dict['test_model']
        ModelHub.register_model_index(
            model_index_path, config_prefix='configs')
        self.assertEqual(ModelHub._models_dict['test_model'].config,
                         osp.abspath(osp.join('configs', 'test_config.py')))

    def test_get_model(self):
        metainfo = ModelHub.get('resnet18_8xb32_in1k')
        self.assertIsInstance(metainfo.weights, str)
        self.assertIsInstance(metainfo.config, Config)


class TestHubAPIs(TestCase):

    def test_list_models(self):
        models_names = list_models()
        self.assertIsInstance(models_names, list)

        models_names = list_models(pattern='swin*in1k')
        for model_name in models_names:
            self.assertTrue(
                model_name.startswith('swin') and 'in1k' in model_name)

    def test_get_model(self):
        model = get_model('mobilenet-v2_8xb32_in1k')
        self.assertIsInstance(model, ImageClassifier)
        self.assertIsInstance(model.backbone, MobileNetV2)

        with patch('mmengine.runner.load_checkpoint') as mock:
            model = get_model('mobilenet-v2_8xb32_in1k', pretrained=True)
            model = get_model('mobilenet-v2_8xb32_in1k', pretrained='test.pth')

            weight = mock.call_args_list[0][0][1]
            self.assertIn('https', weight)
            weight = mock.call_args_list[1][0][1]
            self.assertEqual('test.pth', weight)

        with self.assertRaisesRegex(ValueError, 'Failed to find'):
            get_model('unknown-model')

    def test_init_model(self):
        # test init from config object
        cfg = ModelHub.get('mobilenet-v2_8xb32_in1k').config
        model = init_model(cfg)
        self.assertIsInstance(model, ImageClassifier)
        self.assertIsInstance(model.backbone, MobileNetV2)

        # test init from config file
        cfg = ModelHub._models_dict['mobilenet-v2_8xb32_in1k'].config
        self.assertIsInstance(cfg, str)
        model = init_model(cfg)
        self.assertIsInstance(model, ImageClassifier)
        self.assertIsInstance(model.backbone, MobileNetV2)

        # test modify configs of the model
        model = init_model(cfg, head=dict(num_classes=10))
        self.assertEqual(model.head.num_classes, 10)
