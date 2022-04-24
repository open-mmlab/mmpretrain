# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import tempfile
from unittest import TestCase

import mmcv
import numpy as np

from mmcls.datasets import DATASETS

ASSETS_ROOT = osp.abspath(
    osp.join(osp.dirname(__file__), '../../data/dataset'))


class TestMultiTaskDataset(TestCase):
    DATASET_TYPE = 'MultiTaskDataset'

    DEFAULT_ARGS = dict(
        data_root=ASSETS_ROOT,
        ann_file=osp.join(ASSETS_ROOT, 'multi-task.json'),
        pipeline=[])

    def test_metainfo(self):
        dataset_class = DATASETS.get(self.DATASET_TYPE)

        # Test default behavior
        dataset = dataset_class(**self.DEFAULT_ARGS)
        metainfo = {
            'tasks': [{
                'name': 'gender',
                'categories': ['male', 'female'],
                'type': 'single-label'
            }, {
                'name': 'wear',
                'categories': ['shirt', 'coat', 'jeans', 'pants'],
                'type': 'multi-label'
            }]
        }
        self.assertDictEqual(dataset.metainfo, metainfo)
        self.assertEqual(
            dataset.CLASSES, {
                'gender': ['male', 'female'],
                'wear': ['shirt', 'coat', 'jeans', 'pants']
            })
        self.assertFalse(dataset.test_mode)

        # Test manually specify setting metainfo
        metainfo = {
            'tasks': [{
                'name': 'gender',
                'categories': ['a', 'b'],
                'type': 'single-label'
            }, {
                'name': 'wear',
                'categories': ['c', 'd', 'e'],
                'type': 'multi-label'
            }]
        }
        cfg = {**self.DEFAULT_ARGS, 'metainfo': metainfo}
        dataset = dataset_class(**cfg)
        self.assertDictEqual(dataset.metainfo, metainfo)
        self.assertDictEqual(dataset.CLASSES, {
            'gender': ['a', 'b'],
            'wear': ['c', 'd', 'e']
        })

        # Test invalid metainfo
        cfg = {**self.DEFAULT_ARGS, 'metainfo': ['a', 'b']}
        with self.assertRaisesRegex(AssertionError, "got <class 'list'>"):
            dataset_class(**cfg)

        # Test annotation file without metainfo
        tmpdir = tempfile.TemporaryDirectory()
        new_annotation = mmcv.load(dataset.ann_file)
        del new_annotation['metainfo']
        new_ann_file = osp.abspath(osp.join(tmpdir.name, 'ann_file.json'))
        mmcv.dump(new_annotation, new_ann_file)
        cfg = {**self.DEFAULT_ARGS, 'ann_file': new_ann_file}
        with self.assertRaisesRegex(AssertionError, 'specify the `tasks`'):
            dataset_class(**cfg)

        # Test wrong task type
        cfg['metainfo'] = {'tasks': [['a', 'b'], ['d', 'e']]}
        with self.assertRaisesRegex(AssertionError, 'should be a dict'):
            dataset_class(**cfg)

        # Test incomplete metainfo
        cfg['metainfo'] = {
            'tasks': [{
                'name': 'gender',
                'categories': ['a', 'b'],
            }]
        }
        with self.assertRaisesRegex(AssertionError, 'Missing "type"'):
            dataset_class(**cfg)

        tmpdir.cleanup()

    def test_data_root(self):
        dataset_class = DATASETS.get(self.DATASET_TYPE)

        # Test relative ann_file
        cfg = {**self.DEFAULT_ARGS, 'ann_file': 'multi-task.json'}
        dataset = dataset_class(**cfg)
        self.assertEqual(dataset.data_root, ASSETS_ROOT)
        self.assertEqual(dataset.ann_file,
                         osp.join(dataset.data_root, 'multi-task.json'))

        # Test relative data_prefix
        cfg = {**self.DEFAULT_ARGS, 'data_prefix': 'train'}
        dataset = dataset_class(**cfg)
        self.assertEqual(dataset.data_prefix,
                         osp.join(dataset.data_root, 'train'))

        # Test no data_prefix
        cfg = {**self.DEFAULT_ARGS, 'data_prefix': None}
        dataset = dataset_class(**cfg)
        self.assertEqual(dataset.data_prefix, dataset.data_root)

    def test_parse_data_info(self):
        dataset_class = DATASETS.get(self.DATASET_TYPE)
        dataset = dataset_class(**self.DEFAULT_ARGS)

        data = dataset.parse_data_info({
            'img_path': 'a.jpg',
            'gender_img_label': 0,
            'wear_img_label': [1, 0, 1, 0]
        })
        self.assertDictContainsSubset(
            {
                'img_prefix': ASSETS_ROOT,
                'img_info': {
                    'filename': 'a.jpg'
                }
            }, data)
        np.testing.assert_equal(data['gender_img_label'],
                                np.array(0, dtype=np.int64))
        np.testing.assert_equal(data['wear_img_label'],
                                np.array([1, 0, 1, 0], dtype=np.int8))

        # Test invalid type
        with self.assertRaisesRegex(AssertionError, "got <class 'str'>"):
            dataset.parse_data_info('hi')

        # Test missing path
        with self.assertRaisesRegex(AssertionError, 'have `img_path` field'):
            dataset.parse_data_info({
                'gender_img_label': 0,
                'wear_img_label': [1, 0, 1, 0]
            })

        # Test missing label
        with self.assertRaisesRegex(AssertionError,
                                    'have `gender_img_label` field'):
            dataset.parse_data_info({
                'img_path': 'a.jpg',
                'wear_img_label': [1, 0, 1, 0]
            })

        # Test invalid label type
        with self.assertRaisesRegex(AssertionError, 'a single number'):
            dataset.parse_data_info({
                'img_path': 'a.jpg',
                'gender_img_label': [0, 1],
                'wear_img_label': [1, 0, 1, 0]
            })
        with self.assertRaisesRegex(AssertionError, 'one-hot format'):
            dataset.parse_data_info({
                'img_path': 'a.jpg',
                'gender_img_label': 0,
                'wear_img_label': [1, 2, 3, 4]
            })

    def test_get_cat_ids(self):
        dataset_class = DATASETS.get(self.DATASET_TYPE)
        dataset = dataset_class(**self.DEFAULT_ARGS)

        cat_ids = dataset.get_cat_ids(0)
        self.assertIsInstance(cat_ids, dict)
        self.assertDictEqual(cat_ids, dict(gender=[0], wear=[0, 2]))

    def test_class_to_idx(self):
        dataset_class = DATASETS.get(self.DATASET_TYPE)
        dataset = dataset_class(**self.DEFAULT_ARGS)

        self.assertDictEqual(
            dataset.class_to_idx,
            dict(
                gender={
                    'male': 0,
                    'female': 1
                },
                wear={
                    'shirt': 0,
                    'coat': 1,
                    'jeans': 2,
                    'pants': 3
                }))

    def test_evaluate(self):
        dataset_class = DATASETS.get(self.DATASET_TYPE)
        dataset = dataset_class(**self.DEFAULT_ARGS)

        fake_results = [
            dict(gender=[0.7, 0.3], wear=[1, 0, 1, 0]),
            dict(gender=[0.4, 0.6], wear=[1, 0, 1, 0]),
            dict(gender=[0.1, 0.9], wear=[1, 0, 1, 0])
        ]

        eval_results = dataset.evaluate(
            fake_results,
            metric=dict(
                gender=[
                    'precision', 'recall', 'f1_score', 'support', 'accuracy'
                ],
                wear=['mAP', 'CR', 'OF1']),
            metric_options=dict(gender={'topk': 1}))

        # Test results
        self.assertAlmostEqual(
            eval_results['gender_precision'], (1 / 2 + 1) / 2 * 100.0,
            places=4)
        self.assertAlmostEqual(
            eval_results['gender_recall'], (1 + 1 / 2) / 2 * 100.0, places=4)
        self.assertAlmostEqual(
            eval_results['gender_f1_score'], (2 / 3 + 2 / 3) / 2 * 100.0,
            places=4)
        self.assertEqual(eval_results['gender_support'], 3)
        self.assertAlmostEqual(
            eval_results['gender_accuracy'], 2 / 3 * 100, places=4)
        self.assertAlmostEqual(eval_results['wear_mAP'], 66.67, places=2)
        self.assertAlmostEqual(eval_results['wear_CR'], 50, places=2)
        self.assertAlmostEqual(eval_results['wear_OF1'], 66.67, places=2)

        # test indices
        eval_results = dataset.evaluate(
            fake_results[:2],
            metric=dict(
                gender=[
                    'precision', 'recall', 'f1_score', 'support', 'accuracy'
                ],
                wear=['mAP', 'CR', 'OF1']),
            metric_options=dict(gender={'topk': 1}),
            indices=range(2))
        self.assertAlmostEqual(
            eval_results['gender_precision'], (1 + 0) / 2 * 100.0, places=4)
        self.assertAlmostEqual(
            eval_results['gender_recall'], (1 / 2 + 0) / 2 * 100.0, places=4)
        self.assertAlmostEqual(
            eval_results['gender_f1_score'], (2 / 3 + 0) / 2 * 100.0, places=4)
        self.assertEqual(eval_results['gender_support'], 2)
        self.assertAlmostEqual(
            eval_results['gender_accuracy'], 1 / 2 * 100, places=4)

        self.assertAlmostEqual(eval_results['wear_mAP'], 50, places=2)
        self.assertAlmostEqual(eval_results['wear_CR'], 50, places=2)
        self.assertAlmostEqual(eval_results['wear_OF1'], 100, places=2)

    def test_repr(self):
        dataset_class = DATASETS.get(self.DATASET_TYPE)
        dataset = dataset_class(**self.DEFAULT_ARGS)

        task_doc = ('    For 2 tasks\n'
                    '        gender (2 categories, single-label)\n'
                    '        wear (4 categories, multi-label)')
        self.assertIn(task_doc, repr(dataset))
