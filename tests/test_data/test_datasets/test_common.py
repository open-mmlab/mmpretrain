# Copyright (c) OpenMMLab. All rights reserved.
import os
import os.path as osp
import pickle
import tempfile
from unittest import TestCase
from unittest.mock import patch

import numpy as np
import torch

from mmcls.datasets import DATASETS
from mmcls.datasets import BaseDataset as _BaseDataset
from mmcls.datasets import MultiLabelDataset as _MultiLabelDataset

ASSETS_ROOT = osp.abspath(
    osp.join(osp.dirname(__file__), '../../data/dataset'))


class BaseDataset(_BaseDataset):

    def load_annotations(self):
        pass


class MultiLabelDataset(_MultiLabelDataset):

    def load_annotations(self):
        pass


DATASETS.module_dict['BaseDataset'] = BaseDataset
DATASETS.module_dict['MultiLabelDataset'] = MultiLabelDataset


class TestBaseDataset(TestCase):
    DATASET_TYPE = 'BaseDataset'

    DEFAULT_ARGS = dict(data_prefix='', pipeline=[])

    def test_initialize(self):
        dataset_class = DATASETS.get(self.DATASET_TYPE)

        with patch.object(dataset_class, 'load_annotations'):
            # Test default behavior
            cfg = {**self.DEFAULT_ARGS, 'classes': None, 'ann_file': None}
            dataset = dataset_class(**cfg)
            self.assertEqual(dataset.CLASSES, dataset_class.CLASSES)
            self.assertFalse(dataset.test_mode)
            self.assertIsNone(dataset.ann_file)

            # Test setting classes as a tuple
            cfg = {**self.DEFAULT_ARGS, 'classes': ('bus', 'car')}
            dataset = dataset_class(**cfg)
            self.assertEqual(dataset.CLASSES, ('bus', 'car'))

            # Test setting classes as a tuple
            cfg = {**self.DEFAULT_ARGS, 'classes': ['bus', 'car']}
            dataset = dataset_class(**cfg)
            self.assertEqual(dataset.CLASSES, ['bus', 'car'])

            # Test setting classes through a file
            classes_file = osp.join(ASSETS_ROOT, 'classes.txt')
            cfg = {**self.DEFAULT_ARGS, 'classes': classes_file}
            dataset = dataset_class(**cfg)
            self.assertEqual(dataset.CLASSES, ['bus', 'car'])
            self.assertEqual(dataset.class_to_idx, {'bus': 0, 'car': 1})

            # Test invalid classes
            cfg = {**self.DEFAULT_ARGS, 'classes': dict(classes=1)}
            with self.assertRaisesRegex(ValueError, "type <class 'dict'>"):
                dataset_class(**cfg)

    def test_get_cat_ids(self):
        dataset_class = DATASETS.get(self.DATASET_TYPE)
        fake_ann = [
            dict(
                img_prefix='',
                img_info=dict(),
                gt_label=np.array(0, dtype=np.int64))
        ]

        with patch.object(dataset_class, 'load_annotations') as mock_load:
            mock_load.return_value = fake_ann
            dataset = dataset_class(**self.DEFAULT_ARGS)

        cat_ids = dataset.get_cat_ids(0)
        self.assertIsInstance(cat_ids, list)
        self.assertEqual(len(cat_ids), 1)
        self.assertIsInstance(cat_ids[0], int)

    def test_evaluate(self):
        dataset_class = DATASETS.get(self.DATASET_TYPE)

        fake_ann = [
            dict(gt_label=np.array(0, dtype=np.int64)),
            dict(gt_label=np.array(0, dtype=np.int64)),
            dict(gt_label=np.array(1, dtype=np.int64)),
            dict(gt_label=np.array(2, dtype=np.int64)),
            dict(gt_label=np.array(1, dtype=np.int64)),
            dict(gt_label=np.array(0, dtype=np.int64)),
        ]

        with patch.object(dataset_class, 'load_annotations') as mock_load:
            mock_load.return_value = fake_ann
            dataset = dataset_class(**self.DEFAULT_ARGS)

        fake_results = np.array([
            [0.7, 0.0, 0.3],
            [0.5, 0.2, 0.3],
            [0.4, 0.5, 0.1],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
        ])

        eval_results = dataset.evaluate(
            fake_results,
            metric=['precision', 'recall', 'f1_score', 'support', 'accuracy'],
            metric_options={'topk': 1})

        # Test results
        self.assertAlmostEqual(
            eval_results['precision'], (1 + 1 + 1 / 3) / 3 * 100.0, places=4)
        self.assertAlmostEqual(
            eval_results['recall'], (2 / 3 + 1 / 2 + 1) / 3 * 100.0, places=4)
        self.assertAlmostEqual(
            eval_results['f1_score'], (4 / 5 + 2 / 3 + 1 / 2) / 3 * 100.0,
            places=4)
        self.assertEqual(eval_results['support'], 6)
        self.assertAlmostEqual(eval_results['accuracy'], 4 / 6 * 100, places=4)

        # test indices
        eval_results_ = dataset.evaluate(
            fake_results[:5],
            metric=['precision', 'recall', 'f1_score', 'support', 'accuracy'],
            metric_options={'topk': 1},
            indices=range(5))
        self.assertAlmostEqual(
            eval_results_['precision'], (1 + 1 + 1 / 2) / 3 * 100.0, places=4)
        self.assertAlmostEqual(
            eval_results_['recall'], (1 + 1 / 2 + 1) / 3 * 100.0, places=4)
        self.assertAlmostEqual(
            eval_results_['f1_score'], (1 + 2 / 3 + 2 / 3) / 3 * 100.0,
            places=4)
        self.assertEqual(eval_results_['support'], 5)
        self.assertAlmostEqual(
            eval_results_['accuracy'], 4 / 5 * 100, places=4)

        # test input as tensor
        fake_results_tensor = torch.from_numpy(fake_results)
        eval_results_ = dataset.evaluate(
            fake_results_tensor,
            metric=['precision', 'recall', 'f1_score', 'support', 'accuracy'],
            metric_options={'topk': 1})
        assert eval_results_ == eval_results

        # test thr
        eval_results = dataset.evaluate(
            fake_results,
            metric=['precision', 'recall', 'f1_score', 'accuracy'],
            metric_options={
                'thrs': 0.6,
                'topk': 1
            })

        self.assertAlmostEqual(
            eval_results['precision'], (1 + 0 + 1 / 3) / 3 * 100.0, places=4)
        self.assertAlmostEqual(
            eval_results['recall'], (1 / 3 + 0 + 1) / 3 * 100.0, places=4)
        self.assertAlmostEqual(
            eval_results['f1_score'], (1 / 2 + 0 + 1 / 2) / 3 * 100.0,
            places=4)
        self.assertAlmostEqual(eval_results['accuracy'], 2 / 6 * 100, places=4)

        # thrs must be a number or tuple
        with self.assertRaises(TypeError):
            dataset.evaluate(
                fake_results,
                metric=['precision', 'recall', 'f1_score', 'accuracy'],
                metric_options={
                    'thrs': 'thr',
                    'topk': 1
                })

        # test topk and thr as tuple
        eval_results = dataset.evaluate(
            fake_results,
            metric=['precision', 'recall', 'f1_score', 'accuracy'],
            metric_options={
                'thrs': (0.5, 0.6),
                'topk': (1, 2)
            })
        self.assertEqual(
            {
                'precision_thr_0.50', 'precision_thr_0.60', 'recall_thr_0.50',
                'recall_thr_0.60', 'f1_score_thr_0.50', 'f1_score_thr_0.60',
                'accuracy_top-1_thr_0.50', 'accuracy_top-1_thr_0.60',
                'accuracy_top-2_thr_0.50', 'accuracy_top-2_thr_0.60'
            }, eval_results.keys())

        self.assertIsInstance(eval_results['precision_thr_0.50'], float)
        self.assertIsInstance(eval_results['recall_thr_0.50'], float)
        self.assertIsInstance(eval_results['f1_score_thr_0.50'], float)
        self.assertIsInstance(eval_results['accuracy_top-1_thr_0.50'], float)

        # test topk is tuple while thrs is number
        eval_results = dataset.evaluate(
            fake_results,
            metric='accuracy',
            metric_options={
                'thrs': 0.5,
                'topk': (1, 2)
            })
        self.assertEqual({'accuracy_top-1', 'accuracy_top-2'},
                         eval_results.keys())
        self.assertIsInstance(eval_results['accuracy_top-1'], float)

        # test topk is number while thrs is tuple
        eval_results = dataset.evaluate(
            fake_results,
            metric='accuracy',
            metric_options={
                'thrs': (0.5, 0.6),
                'topk': 1
            })
        self.assertEqual({'accuracy_thr_0.50', 'accuracy_thr_0.60'},
                         eval_results.keys())
        self.assertIsInstance(eval_results['accuracy_thr_0.50'], float)

        # test evaluation results for classes
        eval_results = dataset.evaluate(
            fake_results,
            metric=['precision', 'recall', 'f1_score', 'support'],
            metric_options={'average_mode': 'none'})
        self.assertEqual(eval_results['precision'].shape, (3, ))
        self.assertEqual(eval_results['recall'].shape, (3, ))
        self.assertEqual(eval_results['f1_score'].shape, (3, ))
        self.assertEqual(eval_results['support'].shape, (3, ))

        # the average_mode method must be valid
        with self.assertRaises(ValueError):
            dataset.evaluate(
                fake_results,
                metric=['precision', 'recall', 'f1_score', 'support'],
                metric_options={'average_mode': 'micro'})

        # the metric must be valid for the dataset
        with self.assertRaisesRegex(ValueError,
                                    "{'unknown'} is not supported"):
            dataset.evaluate(fake_results, metric='unknown')


class TestMultiLabelDataset(TestBaseDataset):
    DATASET_TYPE = 'MultiLabelDataset'

    def test_get_cat_ids(self):
        dataset_class = DATASETS.get(self.DATASET_TYPE)
        fake_ann = [
            dict(
                img_prefix='',
                img_info=dict(),
                gt_label=np.array([0, 1, 1, 0], dtype=np.uint8))
        ]

        with patch.object(dataset_class, 'load_annotations') as mock_load:
            mock_load.return_value = fake_ann
            dataset = dataset_class(**self.DEFAULT_ARGS)

        cat_ids = dataset.get_cat_ids(0)
        self.assertIsInstance(cat_ids, list)
        self.assertEqual(len(cat_ids), 2)
        self.assertIsInstance(cat_ids[0], int)
        self.assertEqual(cat_ids, [1, 2])

    def test_evaluate(self):
        dataset_class = DATASETS.get(self.DATASET_TYPE)

        fake_ann = [
            dict(gt_label=np.array([1, 1, 0, -1], dtype=np.int8)),
            dict(gt_label=np.array([1, 1, 0, -1], dtype=np.int8)),
            dict(gt_label=np.array([0, -1, 1, -1], dtype=np.int8)),
            dict(gt_label=np.array([0, 1, 0, -1], dtype=np.int8)),
            dict(gt_label=np.array([0, 1, 0, -1], dtype=np.int8)),
        ]

        with patch.object(dataset_class, 'load_annotations') as mock_load:
            mock_load.return_value = fake_ann
            dataset = dataset_class(**self.DEFAULT_ARGS)

        fake_results = np.array([
            [0.9, 0.8, 0.3, 0.2],
            [0.1, 0.2, 0.2, 0.1],
            [0.7, 0.5, 0.9, 0.3],
            [0.8, 0.1, 0.1, 0.2],
            [0.8, 0.1, 0.1, 0.2],
        ])

        # the metric must be valid for the dataset
        with self.assertRaisesRegex(ValueError,
                                    "{'unknown'} is not supported"):
            dataset.evaluate(fake_results, metric='unknown')

        # only one metric
        eval_results = dataset.evaluate(fake_results, metric='mAP')
        self.assertEqual(eval_results.keys(), {'mAP'})
        self.assertAlmostEqual(eval_results['mAP'], 67.5, places=4)

        # multiple metrics
        eval_results = dataset.evaluate(
            fake_results, metric=['mAP', 'CR', 'OF1'])
        self.assertEqual(eval_results.keys(), {'mAP', 'CR', 'OF1'})
        self.assertAlmostEqual(eval_results['mAP'], 67.50, places=2)
        self.assertAlmostEqual(eval_results['CR'], 43.75, places=2)
        self.assertAlmostEqual(eval_results['OF1'], 42.86, places=2)


class TestCustomDataset(TestBaseDataset):
    DATASET_TYPE = 'CustomDataset'

    def test_load_annotations(self):
        dataset_class = DATASETS.get(self.DATASET_TYPE)

        # test load without ann_file
        cfg = {
            **self.DEFAULT_ARGS,
            'data_prefix': ASSETS_ROOT,
            'ann_file': None,
        }
        dataset = dataset_class(**cfg)
        self.assertEqual(len(dataset), 3)
        self.assertEqual(dataset.CLASSES, ['a', 'b'])  # auto infer classes
        self.assertEqual(
            dataset.data_infos[0], {
                'img_prefix': ASSETS_ROOT,
                'img_info': {
                    'filename': 'a/1.JPG'
                },
                'gt_label': np.array(0)
            })
        self.assertEqual(
            dataset.data_infos[2], {
                'img_prefix': ASSETS_ROOT,
                'img_info': {
                    'filename': 'b/subb/3.jpg'
                },
                'gt_label': np.array(1)
            })

        # test ann_file assertion
        cfg = {
            **self.DEFAULT_ARGS,
            'data_prefix': ASSETS_ROOT,
            'ann_file': ['ann_file.txt'],
        }
        with self.assertRaisesRegex(TypeError, 'must be a str'):
            dataset_class(**cfg)

        # test load with ann_file
        cfg = {
            **self.DEFAULT_ARGS,
            'data_prefix': ASSETS_ROOT,
            'ann_file': osp.join(ASSETS_ROOT, 'ann.txt'),
        }
        dataset = dataset_class(**cfg)
        self.assertEqual(len(dataset), 3)
        # custom dataset won't infer CLASSES from ann_file
        self.assertEqual(dataset.CLASSES, dataset_class.CLASSES)
        self.assertEqual(
            dataset.data_infos[0], {
                'img_prefix': ASSETS_ROOT,
                'img_info': {
                    'filename': 'a/1.JPG'
                },
                'gt_label': np.array(0)
            })
        self.assertEqual(
            dataset.data_infos[2], {
                'img_prefix': ASSETS_ROOT,
                'img_info': {
                    'filename': 'b/subb/2.jpeg'
                },
                'gt_label': np.array(1)
            })

        # test extensions filter
        cfg = {
            **self.DEFAULT_ARGS, 'data_prefix': ASSETS_ROOT,
            'ann_file': None,
            'extensions': ('.txt', )
        }
        with self.assertRaisesRegex(RuntimeError,
                                    'Supported extensions are: .txt'):
            dataset_class(**cfg)

        cfg = {
            **self.DEFAULT_ARGS, 'data_prefix': ASSETS_ROOT,
            'ann_file': None,
            'extensions': ('.jpeg', )
        }
        with self.assertWarnsRegex(UserWarning,
                                   'Supported extensions are: .jpeg'):
            dataset = dataset_class(**cfg)
        self.assertEqual(len(dataset), 1)
        self.assertEqual(
            dataset.data_infos[0], {
                'img_prefix': ASSETS_ROOT,
                'img_info': {
                    'filename': 'b/2.jpeg'
                },
                'gt_label': np.array(1)
            })

        # test classes check
        cfg = {
            **self.DEFAULT_ARGS,
            'data_prefix': ASSETS_ROOT,
            'classes': ['apple', 'banana'],
            'ann_file': None,
        }
        dataset = dataset_class(**cfg)
        self.assertEqual(dataset.CLASSES, ['apple', 'banana'])

        cfg['classes'] = ['apple', 'banana', 'dog']
        with self.assertRaisesRegex(AssertionError,
                                    r"\(2\) doesn't match .* classes \(3\)"):
            dataset_class(**cfg)


class TestImageNet(TestBaseDataset):
    DATASET_TYPE = 'ImageNet'

    def test_load_annotations(self):
        dataset_class = DATASETS.get(self.DATASET_TYPE)

        # test classes number
        cfg = {
            **self.DEFAULT_ARGS,
            'data_prefix': ASSETS_ROOT,
            'ann_file': None,
        }
        with self.assertRaisesRegex(
                AssertionError, r"\(2\) doesn't match .* classes \(1000\)"):
            dataset_class(**cfg)

        # test override classes
        cfg = {
            **self.DEFAULT_ARGS,
            'data_prefix': ASSETS_ROOT,
            'classes': ['cat', 'dog'],
            'ann_file': None,
        }
        dataset = dataset_class(**cfg)
        self.assertEqual(len(dataset), 3)
        self.assertEqual(dataset.CLASSES, ['cat', 'dog'])


class TestImageNet21k(TestBaseDataset):
    DATASET_TYPE = 'ImageNet21k'

    DEFAULT_ARGS = dict(
        data_prefix=ASSETS_ROOT,
        pipeline=[],
        classes=['cat', 'dog'],
        ann_file=osp.join(ASSETS_ROOT, 'ann.txt'),
        serialize_data=False)

    def test_initialize(self):
        super().test_initialize()
        dataset_class = DATASETS.get(self.DATASET_TYPE)

        # The multi_label option is not implemented not.
        cfg = {**self.DEFAULT_ARGS, 'multi_label': True}
        with self.assertRaisesRegex(NotImplementedError, 'not supported'):
            dataset_class(**cfg)

        # Warn about ann_file
        cfg = {**self.DEFAULT_ARGS, 'ann_file': None}
        with self.assertWarnsRegex(UserWarning, 'specify the `ann_file`'):
            dataset_class(**cfg)

        # Warn about classes
        cfg = {**self.DEFAULT_ARGS, 'classes': None}
        with self.assertWarnsRegex(UserWarning, 'specify the `classes`'):
            dataset_class(**cfg)

    def test_load_annotations(self):
        dataset_class = DATASETS.get(self.DATASET_TYPE)

        # Test with serialize_data=False
        cfg = {**self.DEFAULT_ARGS, 'serialize_data': False}
        dataset = dataset_class(**cfg)
        self.assertEqual(len(dataset.data_infos), 3)
        self.assertEqual(len(dataset), 3)
        self.assertEqual(
            dataset[0], {
                'img_prefix': ASSETS_ROOT,
                'img_info': {
                    'filename': 'a/1.JPG'
                },
                'gt_label': np.array(0)
            })
        self.assertEqual(
            dataset[2], {
                'img_prefix': ASSETS_ROOT,
                'img_info': {
                    'filename': 'b/subb/2.jpeg'
                },
                'gt_label': np.array(1)
            })

        # Test with serialize_data=True
        cfg = {**self.DEFAULT_ARGS, 'serialize_data': True}
        dataset = dataset_class(**cfg)
        self.assertEqual(len(dataset.data_infos), 0)  # data_infos is clear.
        self.assertEqual(len(dataset), 3)
        self.assertEqual(
            dataset[0], {
                'img_prefix': ASSETS_ROOT,
                'img_info': {
                    'filename': 'a/1.JPG'
                },
                'gt_label': np.array(0)
            })
        self.assertEqual(
            dataset[2], {
                'img_prefix': ASSETS_ROOT,
                'img_info': {
                    'filename': 'b/subb/2.jpeg'
                },
                'gt_label': np.array(1)
            })


class TestMNIST(TestBaseDataset):
    DATASET_TYPE = 'MNIST'

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()

        tmpdir = tempfile.TemporaryDirectory()
        cls.tmpdir = tmpdir
        data_prefix = tmpdir.name
        cls.DEFAULT_ARGS = dict(data_prefix=data_prefix, pipeline=[])

        dataset_class = DATASETS.get(cls.DATASET_TYPE)

        def rm_suffix(s):
            return s[:s.rfind('.')]

        train_image_file = osp.join(
            data_prefix,
            rm_suffix(dataset_class.resources['train_image_file'][0]))
        train_label_file = osp.join(
            data_prefix,
            rm_suffix(dataset_class.resources['train_label_file'][0]))
        test_image_file = osp.join(
            data_prefix,
            rm_suffix(dataset_class.resources['test_image_file'][0]))
        test_label_file = osp.join(
            data_prefix,
            rm_suffix(dataset_class.resources['test_label_file'][0]))
        cls.fake_img = np.random.randint(0, 255, size=(28, 28), dtype=np.uint8)
        cls.fake_label = np.random.randint(0, 10, size=(1, ), dtype=np.uint8)

        for file in [train_image_file, test_image_file]:
            magic = b'\x00\x00\x08\x03'  # num_dims = 3, type = uint8
            head = b'\x00\x00\x00\x01' + b'\x00\x00\x00\x1c' * 2  # (1, 28, 28)
            data = magic + head + cls.fake_img.flatten().tobytes()
            with open(file, 'wb') as f:
                f.write(data)

        for file in [train_label_file, test_label_file]:
            magic = b'\x00\x00\x08\x01'  # num_dims = 3, type = uint8
            head = b'\x00\x00\x00\x01'  # (1, )
            data = magic + head + cls.fake_label.tobytes()
            with open(file, 'wb') as f:
                f.write(data)

    def test_load_annotations(self):
        dataset_class = DATASETS.get(self.DATASET_TYPE)

        with patch.object(dataset_class, 'download'):
            # Test default behavior
            dataset = dataset_class(**self.DEFAULT_ARGS)
            self.assertEqual(len(dataset), 1)

            data_info = dataset[0]
            np.testing.assert_equal(data_info['img'], self.fake_img)
            np.testing.assert_equal(data_info['gt_label'], self.fake_label)

    @classmethod
    def tearDownClass(cls):
        cls.tmpdir.cleanup()


class TestCIFAR10(TestBaseDataset):
    DATASET_TYPE = 'CIFAR10'

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()

        tmpdir = tempfile.TemporaryDirectory()
        cls.tmpdir = tmpdir
        data_prefix = tmpdir.name
        cls.DEFAULT_ARGS = dict(data_prefix=data_prefix, pipeline=[])

        dataset_class = DATASETS.get(cls.DATASET_TYPE)
        base_folder = osp.join(data_prefix, dataset_class.base_folder)
        os.mkdir(base_folder)

        cls.fake_imgs = np.random.randint(
            0, 255, size=(6, 3 * 32 * 32), dtype=np.uint8)
        cls.fake_labels = np.random.randint(0, 10, size=(6, ))
        cls.fake_classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

        batch1 = dict(
            data=cls.fake_imgs[:2], labels=cls.fake_labels[:2].tolist())
        with open(osp.join(base_folder, 'data_batch_1'), 'wb') as f:
            f.write(pickle.dumps(batch1))

        batch2 = dict(
            data=cls.fake_imgs[2:4], labels=cls.fake_labels[2:4].tolist())
        with open(osp.join(base_folder, 'data_batch_2'), 'wb') as f:
            f.write(pickle.dumps(batch2))

        test_batch = dict(
            data=cls.fake_imgs[4:], labels=cls.fake_labels[4:].tolist())
        with open(osp.join(base_folder, 'test_batch'), 'wb') as f:
            f.write(pickle.dumps(test_batch))

        meta = {dataset_class.meta['key']: cls.fake_classes}
        meta_filename = dataset_class.meta['filename']
        with open(osp.join(base_folder, meta_filename), 'wb') as f:
            f.write(pickle.dumps(meta))

        dataset_class.train_list = [['data_batch_1', None],
                                    ['data_batch_2', None]]
        dataset_class.test_list = [['test_batch', None]]
        dataset_class.meta['md5'] = None

    def test_load_annotations(self):
        dataset_class = DATASETS.get(self.DATASET_TYPE)

        # Test default behavior
        dataset = dataset_class(**self.DEFAULT_ARGS)
        self.assertEqual(len(dataset), 4)
        self.assertEqual(dataset.CLASSES, self.fake_classes)

        data_info = dataset[0]
        fake_img = self.fake_imgs[0].reshape(3, 32, 32).transpose(1, 2, 0)
        np.testing.assert_equal(data_info['img'], fake_img)
        np.testing.assert_equal(data_info['gt_label'], self.fake_labels[0])

        # Test with test_mode=True
        cfg = {**self.DEFAULT_ARGS, 'test_mode': True}
        dataset = dataset_class(**cfg)
        self.assertEqual(len(dataset), 2)

        data_info = dataset[0]
        fake_img = self.fake_imgs[4].reshape(3, 32, 32).transpose(1, 2, 0)
        np.testing.assert_equal(data_info['img'], fake_img)
        np.testing.assert_equal(data_info['gt_label'], self.fake_labels[4])

    @classmethod
    def tearDownClass(cls):
        cls.tmpdir.cleanup()


class TestCIFAR100(TestCIFAR10):
    DATASET_TYPE = 'CIFAR100'


class TestVOC(TestMultiLabelDataset):
    DATASET_TYPE = 'VOC'

    DEFAULT_ARGS = dict(data_prefix='VOC2007', pipeline=[])


class TestCUB(TestBaseDataset):
    DATASET_TYPE = 'CUB'

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()

        tmpdir = tempfile.TemporaryDirectory()
        cls.tmpdir = tmpdir
        cls.data_prefix = tmpdir.name
        cls.ann_file = osp.join(cls.data_prefix, 'ann_file.txt')
        cls.image_class_labels_file = osp.join(cls.data_prefix, 'classes.txt')
        cls.train_test_split_file = osp.join(cls.data_prefix, 'split.txt')
        cls.train_test_split_file2 = osp.join(cls.data_prefix, 'split2.txt')
        cls.DEFAULT_ARGS = dict(
            data_prefix=cls.data_prefix,
            pipeline=[],
            ann_file=cls.ann_file,
            image_class_labels_file=cls.image_class_labels_file,
            train_test_split_file=cls.train_test_split_file)

        with open(cls.ann_file, 'w') as f:
            f.write('\n'.join([
                '1 1.txt',
                '2 2.txt',
                '3 3.txt',
            ]))

        with open(cls.image_class_labels_file, 'w') as f:
            f.write('\n'.join([
                '1 2',
                '2 3',
                '3 1',
            ]))

        with open(cls.train_test_split_file, 'w') as f:
            f.write('\n'.join([
                '1 0',
                '2 1',
                '3 1',
            ]))

        with open(cls.train_test_split_file2, 'w') as f:
            f.write('\n'.join([
                '1 0',
                '2 1',
            ]))

    def test_load_annotations(self):
        dataset_class = DATASETS.get(self.DATASET_TYPE)

        # Test default behavior
        dataset = dataset_class(**self.DEFAULT_ARGS)
        self.assertEqual(len(dataset), 2)
        self.assertEqual(dataset.CLASSES, dataset_class.CLASSES)

        data_info = dataset[0]
        np.testing.assert_equal(data_info['img_prefix'], self.data_prefix)
        np.testing.assert_equal(data_info['img_info'], {'filename': '2.txt'})
        np.testing.assert_equal(data_info['gt_label'], 3 - 1)

        # Test with test_mode=True
        cfg = {**self.DEFAULT_ARGS, 'test_mode': True}
        dataset = dataset_class(**cfg)
        self.assertEqual(len(dataset), 1)

        data_info = dataset[0]
        np.testing.assert_equal(data_info['img_prefix'], self.data_prefix)
        np.testing.assert_equal(data_info['img_info'], {'filename': '1.txt'})
        np.testing.assert_equal(data_info['gt_label'], 2 - 1)

        # Test if the numbers of line are not match
        cfg = {
            **self.DEFAULT_ARGS, 'train_test_split_file':
            self.train_test_split_file2
        }
        with self.assertRaisesRegex(AssertionError, 'should have same length'):
            dataset_class(**cfg)

    @classmethod
    def tearDownClass(cls):
        cls.tmpdir.cleanup()
