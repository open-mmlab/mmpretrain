# Copyright (c) OpenMMLab. All rights reserved.
import os
import os.path as osp
import pickle
import sys
import tempfile
from unittest import TestCase
from unittest.mock import MagicMock, call, patch

import mat4py
import numpy as np
from mmengine.logging import MMLogger

from mmpretrain.registry import DATASETS, TRANSFORMS

ASSETS_ROOT = osp.abspath(osp.join(osp.dirname(__file__), '../data/dataset'))


class TestBaseDataset(TestCase):
    DATASET_TYPE = 'BaseDataset'

    DEFAULT_ARGS = dict(data_root=ASSETS_ROOT, ann_file='ann.json')

    def test_initialize(self):
        dataset_class = DATASETS.get(self.DATASET_TYPE)

        # Test loading metainfo from ann_file
        cfg = {**self.DEFAULT_ARGS, 'metainfo': None, 'classes': None}
        dataset = dataset_class(**cfg)
        self.assertEqual(
            dataset.CLASSES,
            dataset_class.METAINFO.get('classes', ('first', 'second')))
        self.assertFalse(dataset.test_mode)

        # Test overriding metainfo by `metainfo` argument
        cfg = {**self.DEFAULT_ARGS, 'metainfo': {'classes': ('bus', 'car')}}
        dataset = dataset_class(**cfg)
        self.assertEqual(dataset.CLASSES, ('bus', 'car'))

        # Test overriding metainfo by `classes` argument
        cfg = {**self.DEFAULT_ARGS, 'classes': ['bus', 'car']}
        dataset = dataset_class(**cfg)
        self.assertEqual(dataset.CLASSES, ('bus', 'car'))

        classes_file = osp.join(ASSETS_ROOT, 'classes.txt')
        cfg = {**self.DEFAULT_ARGS, 'classes': classes_file}
        dataset = dataset_class(**cfg)
        self.assertEqual(dataset.CLASSES, ('bus', 'car'))
        self.assertEqual(dataset.class_to_idx, {'bus': 0, 'car': 1})

        # Test invalid classes
        cfg = {**self.DEFAULT_ARGS, 'classes': dict(classes=1)}
        with self.assertRaisesRegex(ValueError, "type <class 'dict'>"):
            dataset_class(**cfg)

    def test_get_cat_ids(self):
        dataset_class = DATASETS.get(self.DATASET_TYPE)
        dataset = dataset_class(**self.DEFAULT_ARGS)

        cat_ids = dataset.get_cat_ids(0)
        self.assertIsInstance(cat_ids, list)
        self.assertEqual(len(cat_ids), 1)
        self.assertIsInstance(cat_ids[0], int)

    def test_repr(self):
        dataset_class = DATASETS.get(self.DATASET_TYPE)
        cfg = {**self.DEFAULT_ARGS, 'lazy_init': True}
        dataset = dataset_class(**cfg)

        head = 'Dataset ' + dataset.__class__.__name__
        self.assertIn(head, repr(dataset))

        if dataset.CLASSES is not None:
            num_classes = len(dataset.CLASSES)
            self.assertIn(f'Number of categories: \t{num_classes}',
                          repr(dataset))

        self.assertIn('Haven\'t been initialized', repr(dataset))
        dataset.full_init()
        self.assertIn(f'Number of samples: \t{len(dataset)}', repr(dataset))

        TRANSFORMS.register_module(name='test_mock', module=MagicMock)
        cfg = {**self.DEFAULT_ARGS, 'pipeline': [dict(type='test_mock')]}
        dataset = dataset_class(**cfg)
        self.assertIn('With transforms', repr(dataset))
        del TRANSFORMS.module_dict['test_mock']

    def test_extra_repr(self):
        dataset_class = DATASETS.get(self.DATASET_TYPE)
        cfg = {**self.DEFAULT_ARGS, 'lazy_init': True}
        dataset = dataset_class(**cfg)

        self.assertIn(f'Annotation file: \t{dataset.ann_file}', repr(dataset))
        self.assertIn(f'Prefix of images: \t{dataset.img_prefix}',
                      repr(dataset))


class TestCustomDataset(TestBaseDataset):
    DATASET_TYPE = 'CustomDataset'

    DEFAULT_ARGS = dict(data_root=ASSETS_ROOT, ann_file='ann.txt')

    def test_initialize(self):
        dataset_class = DATASETS.get(self.DATASET_TYPE)

        # Test overriding metainfo by `metainfo` argument
        cfg = {**self.DEFAULT_ARGS, 'metainfo': {'classes': ('bus', 'car')}}
        dataset = dataset_class(**cfg)
        self.assertEqual(dataset.CLASSES, ('bus', 'car'))

        # Test overriding metainfo by `classes` argument
        cfg = {**self.DEFAULT_ARGS, 'classes': ['bus', 'car']}
        dataset = dataset_class(**cfg)
        self.assertEqual(dataset.CLASSES, ('bus', 'car'))

        classes_file = osp.join(ASSETS_ROOT, 'classes.txt')
        cfg = {**self.DEFAULT_ARGS, 'classes': classes_file}
        dataset = dataset_class(**cfg)
        self.assertEqual(dataset.CLASSES, ('bus', 'car'))
        self.assertEqual(dataset.class_to_idx, {'bus': 0, 'car': 1})

        # Test invalid classes
        cfg = {**self.DEFAULT_ARGS, 'classes': dict(classes=1)}
        with self.assertRaisesRegex(ValueError, "type <class 'dict'>"):
            dataset_class(**cfg)

    def test_load_data_list(self):
        dataset_class = DATASETS.get(self.DATASET_TYPE)

        # test load without ann_file
        cfg = {
            **self.DEFAULT_ARGS,
            'data_prefix': ASSETS_ROOT,
            'ann_file': '',
        }
        dataset = dataset_class(**cfg)
        self.assertEqual(len(dataset), 3)
        self.assertEqual(dataset.CLASSES, ('a', 'b'))  # auto infer classes
        self.assertGreaterEqual(
            dataset.get_data_info(0).items(), {
                'img_path': osp.join(ASSETS_ROOT, 'a', '1.JPG'),
                'gt_label': 0
            }.items())
        self.assertGreaterEqual(
            dataset.get_data_info(2).items(), {
                'img_path': osp.join(ASSETS_ROOT, 'b', 'subb', '3.jpg'),
                'gt_label': 1
            }.items())

        # test load without ann_file and without labels
        # (no specific folder structures)
        cfg = {
            **self.DEFAULT_ARGS,
            'data_prefix': ASSETS_ROOT,
            'ann_file': '',
            'with_label': False,
        }
        dataset = dataset_class(**cfg)
        self.assertEqual(len(dataset), 4)
        self.assertIsNone(dataset.CLASSES, None)
        self.assertGreaterEqual(
            dataset.get_data_info(0).items(), {
                'img_path': osp.join(ASSETS_ROOT, '3.jpeg'),
            }.items())
        self.assertGreaterEqual(
            dataset.get_data_info(1).items(), {
                'img_path': osp.join(ASSETS_ROOT, 'a', '1.JPG'),
            }.items())
        self.assertGreaterEqual(
            dataset.get_data_info(3).items(), {
                'img_path': osp.join(ASSETS_ROOT, 'b', 'subb', '3.jpg'),
            }.items())

        # test ann_file assertion
        cfg = {
            **self.DEFAULT_ARGS,
            'data_prefix': ASSETS_ROOT,
            'ann_file': ['ann_file.txt'],
        }
        with self.assertRaisesRegex(TypeError, 'expected str'):
            dataset_class(**cfg)

        # test load with ann_file
        cfg = {
            **self.DEFAULT_ARGS,
            'data_root': ASSETS_ROOT,
            'ann_file': 'ann.txt',
        }
        dataset = dataset_class(**cfg)
        self.assertEqual(len(dataset), 3)
        # custom dataset won't infer CLASSES from ann_file
        self.assertIsNone(dataset.CLASSES, None)
        self.assertGreaterEqual(
            dataset.get_data_info(0).items(), {
                'img_path': osp.join(ASSETS_ROOT, 'a/1.JPG'),
                'gt_label': 0,
            }.items())
        self.assertGreaterEqual(
            dataset.get_data_info(2).items(), {
                'img_path': osp.join(ASSETS_ROOT, 'b/subb/3.jpg'),
                'gt_label': 1
            }.items())
        np.testing.assert_equal(dataset.get_gt_labels(), np.array([0, 1, 1]))

        # test load with absolute ann_file
        cfg = {
            **self.DEFAULT_ARGS,
            'data_root': '',
            'data_prefix': '',
            'ann_file': osp.join(ASSETS_ROOT, 'ann.txt'),
        }
        dataset = dataset_class(**cfg)
        self.assertEqual(len(dataset), 3)
        # custom dataset won't infer CLASSES from ann_file
        self.assertIsNone(dataset.CLASSES, None)
        self.assertGreaterEqual(
            dataset.get_data_info(0).items(), {
                'img_path': 'a/1.JPG',
                'gt_label': 0,
            }.items())
        self.assertGreaterEqual(
            dataset.get_data_info(2).items(), {
                'img_path': 'b/subb/3.jpg',
                'gt_label': 1
            }.items())

        # test load with absolute ann_file and without label
        cfg = {
            **self.DEFAULT_ARGS,
            'data_root': '',
            'data_prefix': '',
            'ann_file': osp.join(ASSETS_ROOT, 'ann_without_labels.txt'),
            'with_label': False,
        }
        dataset = dataset_class(**cfg)
        self.assertEqual(len(dataset), 3)
        # custom dataset won't infer CLASSES from ann_file
        self.assertIsNone(dataset.CLASSES, None)
        self.assertGreaterEqual(
            dataset.get_data_info(0).items(), {
                'img_path': 'a/1.JPG',
            }.items())
        self.assertGreaterEqual(
            dataset.get_data_info(2).items(), {
                'img_path': 'b/subb/3.jpg',
            }.items())

        # test extensions filter
        cfg = {
            **self.DEFAULT_ARGS, 'data_prefix': dict(img_path=ASSETS_ROOT),
            'ann_file': '',
            'extensions': ('.txt', )
        }
        with self.assertRaisesRegex(RuntimeError,
                                    'Supported extensions are: .txt'):
            dataset_class(**cfg)

        cfg = {
            **self.DEFAULT_ARGS, 'data_prefix': ASSETS_ROOT,
            'ann_file': '',
            'extensions': ('.jpeg', )
        }
        logger = MMLogger.get_current_instance()
        with self.assertLogs(logger, 'WARN') as log:
            dataset = dataset_class(**cfg)
        self.assertIn('Supported extensions are: .jpeg', log.output[0])
        self.assertEqual(len(dataset), 1)
        self.assertGreaterEqual(
            dataset.get_data_info(0).items(), {
                'img_path': osp.join(ASSETS_ROOT, 'b', '2.jpeg'),
                'gt_label': 1
            }.items())

        # test classes check
        cfg = {
            **self.DEFAULT_ARGS,
            'data_prefix': ASSETS_ROOT,
            'classes': ('apple', 'banana'),
            'ann_file': '',
        }
        dataset = dataset_class(**cfg)
        self.assertEqual(dataset.CLASSES, ('apple', 'banana'))

        cfg['classes'] = ['apple', 'banana', 'dog']
        with self.assertRaisesRegex(AssertionError,
                                    r"\(2\) doesn't match .* classes \(3\)"):
            dataset_class(**cfg)


class TestImageNet(TestCustomDataset):
    DATASET_TYPE = 'ImageNet'

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()

        tmpdir = tempfile.TemporaryDirectory()
        cls.tmpdir = tmpdir
        cls.root = tmpdir.name
        cls.meta_folder = 'meta'
        cls.train_file = 'train.txt'
        cls.val_file = 'val.txt'
        cls.test_file = 'test.txt'
        cls.categories = ['cat', 'dog']

        os.mkdir(osp.join(cls.root, cls.meta_folder))

        cls.DEFAULT_ARGS = dict(data_root=cls.root, split='train')

        with open(osp.join(cls.root, cls.meta_folder, cls.train_file),
                  'w') as f:
            f.write('\n'.join([
                '1.jpg 0',
                '2.jpg 1',
                '3.jpg 1',
            ]))

        with open(osp.join(cls.root, cls.meta_folder, cls.val_file), 'w') as f:
            f.write('\n'.join([
                '11.jpg 0',
                '22.jpg 1',
            ]))

        with open(osp.join(cls.root, cls.meta_folder, cls.test_file),
                  'w') as f:
            f.write('\n'.join([
                'aa.jpg',
                'bb.jpg',
            ]))

    def test_initialize(self):
        super().test_initialize()

        dataset_class = DATASETS.get(self.DATASET_TYPE)

        # Test invalid split
        with self.assertRaisesRegex(AssertionError, 'The split must be'):
            cfg = {**self.DEFAULT_ARGS}
            cfg['split'] = 'unknown'
            dataset_class(**cfg)

        # Test valid splits
        splits = ['train', 'val']
        for split in splits:
            cfg = {**self.DEFAULT_ARGS}
            cfg['split'] = split
            cfg['classes'] = self.categories
            dataset = dataset_class(**cfg)
            self.assertEqual(dataset.data_root, self.root)

        # Test split="test"
        cfg = {**self.DEFAULT_ARGS}
        cfg['split'] = 'test'
        logger = MMLogger.get_current_instance()
        with self.assertLogs(logger, 'INFO') as log:
            dataset = dataset_class(**cfg)
            self.assertFalse(dataset.with_label)
        self.assertIn('Since the ImageNet1k test set', log.output[0])

    def test_load_data_list(self):
        dataset_class = DATASETS.get(self.DATASET_TYPE)

        # Test default behavior
        dataset = dataset_class(**self.DEFAULT_ARGS)
        self.assertEqual(len(dataset), 3)

        data_info = dataset[0]
        self.assertEqual(data_info['img_path'],
                         osp.join(self.root, 'train', '1.jpg'))
        self.assertEqual(data_info['gt_label'], 0)

        # Test split="val"
        cfg = {**self.DEFAULT_ARGS, 'split': 'val'}
        dataset = dataset_class(**cfg)
        self.assertEqual(len(dataset), 2)

        data_info = dataset[0]
        self.assertEqual(data_info['img_path'],
                         osp.join(self.root, 'val', '11.jpg'))
        self.assertEqual(data_info['gt_label'], 0)

        # Test split="test"
        cfg = {**self.DEFAULT_ARGS, 'split': 'test'}
        dataset = dataset_class(**cfg)
        self.assertEqual(len(dataset), 2)

        data_info = dataset[0]
        self.assertEqual(data_info['img_path'],
                         osp.join(self.root, 'test', 'aa.jpg'))

        # test override classes
        cfg = {
            **self.DEFAULT_ARGS,
            'classes': ['cat', 'dog'],
        }
        dataset = dataset_class(**cfg)
        self.assertEqual(len(dataset), 3)
        self.assertEqual(dataset.CLASSES, ('cat', 'dog'))

    def test_extra_repr(self):
        dataset_class = DATASETS.get(self.DATASET_TYPE)
        cfg = {**self.DEFAULT_ARGS}
        dataset = dataset_class(**cfg)

        self.assertIn(f'Root of dataset: \t{dataset.data_root}', repr(dataset))


class TestImageNet21k(TestCustomDataset):
    DATASET_TYPE = 'ImageNet21k'

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()

        tmpdir = tempfile.TemporaryDirectory()
        cls.tmpdir = tmpdir
        cls.root = tmpdir.name
        cls.meta_folder = 'meta'
        cls.train_file = 'train.txt'

        os.mkdir(osp.join(cls.root, cls.meta_folder))

        with open(osp.join(cls.root, cls.meta_folder, cls.train_file),
                  'w') as f:
            f.write('\n'.join([
                'cat/a.jpg 0',
                'cat/b.jpg 0',
                'dog/a.jpg 1',
                'dog/b.jpg 1',
            ]))

        cls.DEFAULT_ARGS = dict(
            data_root=cls.root,
            classes=['cat', 'dog'],
            ann_file='meta/train.txt')

    def test_initialize(self):
        super().test_initialize()

        dataset_class = DATASETS.get(self.DATASET_TYPE)

        # Test invalid split
        with self.assertRaisesRegex(AssertionError, 'The split must be'):
            cfg = {**self.DEFAULT_ARGS}
            cfg['split'] = 'unknown'
            dataset_class(**cfg)

        # Test valid splits
        cfg = {**self.DEFAULT_ARGS}
        cfg['split'] = 'train'
        dataset = dataset_class(**cfg)
        self.assertEqual(dataset.split, 'train')
        self.assertEqual(dataset.data_root, self.root)

    def test_load_data_list(self):
        dataset_class = DATASETS.get(self.DATASET_TYPE)

        # The multi_label option is not implemented not.
        cfg = {**self.DEFAULT_ARGS, 'multi_label': True}
        with self.assertRaisesRegex(NotImplementedError, 'not supported'):
            dataset_class(**cfg)

        # Warn about ann_file
        cfg = {**self.DEFAULT_ARGS, 'ann_file': '', 'lazy_init': True}
        ann_path = osp.join(self.root, self.meta_folder, self.train_file)
        os.rename(ann_path, ann_path + 'copy')
        logger = MMLogger.get_current_instance()
        with self.assertLogs(logger, 'INFO') as log:
            dataset_class(**cfg)
        self.assertIn('specify the `ann_file`', log.output[0])
        os.rename(ann_path + 'copy', ann_path)

        # Warn about classes
        cfg = {**self.DEFAULT_ARGS, 'classes': None}
        with self.assertLogs(logger, 'WARN') as log:
            dataset_class(**cfg)
        self.assertIn('specify the `classes`', log.output[0])

        # Test split='train'
        cfg = {**self.DEFAULT_ARGS, 'split': 'train', 'classes': None}
        dataset = dataset_class(**self.DEFAULT_ARGS)
        self.assertEqual(len(dataset), 4)


class TestPlaces205(TestCustomDataset):
    DATASET_TYPE = 'Places205'

    DEFAULT_ARGS = dict(data_root=ASSETS_ROOT, ann_file='ann.txt')

    def test_load_data_list(self):
        dataset_class = DATASETS.get(self.DATASET_TYPE)

        # test classes number
        cfg = {
            **self.DEFAULT_ARGS,
            'data_prefix': ASSETS_ROOT,
            'ann_file': '',
        }
        with self.assertRaisesRegex(AssertionError,
                                    r"\(2\) doesn't match .* classes \(205\)"):
            dataset_class(**cfg)

        # test override classes
        cfg = {
            **self.DEFAULT_ARGS,
            'data_prefix': ASSETS_ROOT,
            'classes': ['cat', 'dog'],
            'ann_file': '',
        }
        dataset = dataset_class(**cfg)
        self.assertEqual(len(dataset), 3)
        self.assertEqual(dataset.CLASSES, ('cat', 'dog'))


class TestCIFAR10(TestBaseDataset):
    DATASET_TYPE = 'CIFAR10'

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()

        tmpdir = tempfile.TemporaryDirectory()
        cls.tmpdir = tmpdir
        cls.root = tmpdir.name
        cls.DEFAULT_ARGS = dict(data_root=cls.root, split='train')

        dataset_class = DATASETS.get(cls.DATASET_TYPE)
        base_folder = osp.join(cls.root, dataset_class.base_folder)
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
            data=cls.fake_imgs[4:], fine_labels=cls.fake_labels[4:].tolist())
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

    def test_initialize(self):
        dataset_class = DATASETS.get(self.DATASET_TYPE)

        # Test with invalid split
        with self.assertRaisesRegex(AssertionError, 'The split must be'):
            cfg = {**self.DEFAULT_ARGS}
            cfg['split'] = 'unknown'
            dataset_class(**cfg)

        # Test with valid split
        splits = ['train', 'test']
        test_modes = [False, True]

        for split in splits:
            for test_mode in test_modes:
                cfg = {**self.DEFAULT_ARGS}
                cfg['split'] = split
                cfg['test_mode'] = test_mode

                if split == 'train' and test_mode:
                    logger = MMLogger.get_current_instance()
                    with self.assertLogs(logger, 'WARN') as log:
                        dataset = dataset_class(**cfg)
                        self.assertEqual(dataset.split, split)
                        self.assertEqual(dataset.test_mode, test_mode)
                        self.assertEqual(dataset.data_root, self.root)
                    self.assertIn('training set will be used', log.output[0])
                else:
                    dataset = dataset_class(**cfg)
                    self.assertEqual(dataset.split, split)
                    self.assertEqual(dataset.test_mode, test_mode)
                    self.assertEqual(dataset.data_root, self.root)

        # Test without dataset path
        with self.assertRaisesRegex(RuntimeError, 'specify the dataset path'):
            dataset = dataset_class()

        # Test overriding metainfo by `metainfo` argument
        cfg = {**self.DEFAULT_ARGS, 'metainfo': {'classes': ('bus', 'car')}}
        dataset = dataset_class(**cfg)
        self.assertEqual(dataset.CLASSES, ('bus', 'car'))

        # Test overriding metainfo by `classes` argument
        cfg = {**self.DEFAULT_ARGS, 'classes': ['bus', 'car']}
        dataset = dataset_class(**cfg)
        self.assertEqual(dataset.CLASSES, ('bus', 'car'))

        classes_file = osp.join(ASSETS_ROOT, 'classes.txt')
        cfg = {**self.DEFAULT_ARGS, 'classes': classes_file}
        dataset = dataset_class(**cfg)
        self.assertEqual(dataset.CLASSES, ('bus', 'car'))
        self.assertEqual(dataset.class_to_idx, {'bus': 0, 'car': 1})

        # Test invalid classes
        cfg = {**self.DEFAULT_ARGS, 'classes': dict(classes=1)}
        with self.assertRaisesRegex(ValueError, "type <class 'dict'>"):
            dataset_class(**cfg)

    def test_load_data_list(self):
        dataset_class = DATASETS.get(self.DATASET_TYPE)

        # Test default behavior
        dataset = dataset_class(**self.DEFAULT_ARGS)
        self.assertEqual(len(dataset), 4)
        self.assertEqual(dataset.CLASSES, dataset_class.METAINFO['classes'])

        data_info = dataset[0]
        fake_img = self.fake_imgs[0].reshape(3, 32, 32).transpose(1, 2, 0)
        np.testing.assert_equal(data_info['img'], fake_img)
        np.testing.assert_equal(data_info['gt_label'], self.fake_labels[0])

        # Test with split='test'
        cfg = {**self.DEFAULT_ARGS, 'split': 'test'}
        dataset = dataset_class(**cfg)
        self.assertEqual(len(dataset), 2)

        data_info = dataset[0]
        fake_img = self.fake_imgs[4].reshape(3, 32, 32).transpose(1, 2, 0)
        np.testing.assert_equal(data_info['img'], fake_img)
        np.testing.assert_equal(data_info['gt_label'], self.fake_labels[4])

        # Test load meta
        cfg = {**self.DEFAULT_ARGS, 'lazy_init': True}
        dataset = dataset_class(**cfg)
        dataset._metainfo = {}
        dataset.full_init()
        self.assertEqual(dataset.CLASSES, self.fake_classes)

        cfg = {**self.DEFAULT_ARGS, 'lazy_init': True}
        dataset = dataset_class(**cfg)
        dataset._metainfo = {}
        dataset.meta['filename'] = 'invalid'
        with self.assertRaisesRegex(RuntimeError, 'not found or corrupted'):
            dataset.full_init()

        # Test automatically download
        with patch('mmpretrain.datasets.cifar.download_and_extract_archive'
                   ) as mock:
            cfg = {**self.DEFAULT_ARGS, 'lazy_init': True, 'split': 'test'}
            dataset = dataset_class(**cfg)
            dataset.test_list = [['invalid_batch', None]]
            with self.assertRaisesRegex(AssertionError, 'Download failed'):
                dataset.full_init()
            mock.assert_called_once_with(
                dataset.url,
                dataset.data_prefix['root'],
                filename=dataset.filename,
                md5=dataset.tgz_md5)

        with self.assertRaisesRegex(RuntimeError, '`download=True`'):
            cfg = {
                **self.DEFAULT_ARGS, 'lazy_init': True,
                'split': 'test',
                'download': False
            }
            dataset = dataset_class(**cfg)
            dataset.test_list = [['test_batch', 'invalid_md5']]
            dataset.full_init()

        # Test different backend
        cfg = {
            **self.DEFAULT_ARGS, 'lazy_init': True,
            'data_prefix': 'http://openmmlab/cifar'
        }
        dataset = dataset_class(**cfg)
        dataset._check_integrity = MagicMock(return_value=False)
        with self.assertRaisesRegex(RuntimeError, 'http://openmmlab/cifar'):
            dataset.full_init()

    def test_extra_repr(self):
        dataset_class = DATASETS.get(self.DATASET_TYPE)
        cfg = {**self.DEFAULT_ARGS, 'lazy_init': True}
        dataset = dataset_class(**cfg)

        self.assertIn(f"Prefix of data: \t{dataset.data_prefix['root']}",
                      repr(dataset))

    @classmethod
    def tearDownClass(cls):
        cls.tmpdir.cleanup()


class TestCIFAR100(TestCIFAR10):
    DATASET_TYPE = 'CIFAR100'


class TestMultiLabelDataset(TestBaseDataset):
    DATASET_TYPE = 'MultiLabelDataset'

    DEFAULT_ARGS = dict(data_root=ASSETS_ROOT, ann_file='multi_label_ann.json')

    def test_get_cat_ids(self):
        dataset_class = DATASETS.get(self.DATASET_TYPE)
        cfg = {**self.DEFAULT_ARGS}
        dataset = dataset_class(**cfg)

        cat_ids = dataset.get_cat_ids(0)
        self.assertTrue(cat_ids, [0])

        cat_ids = dataset.get_cat_ids(1)
        self.assertTrue(cat_ids, [1])

        cat_ids = dataset.get_cat_ids(1)
        self.assertTrue(cat_ids, [0, 1])


class TestVOC(TestBaseDataset):
    DATASET_TYPE = 'VOC'

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()

        tmpdir = tempfile.TemporaryDirectory()
        cls.tmpdir = tmpdir
        data_root = tmpdir.name

        cls.DEFAULT_ARGS = dict(data_root=data_root, split='trainval')

        cls.image_folder = osp.join(data_root, 'JPEGImages')
        cls.ann_folder = osp.join(data_root, 'Annotations')
        cls.image_set_folder = osp.join(data_root, 'ImageSets', 'Main')
        os.makedirs(cls.image_set_folder)
        os.mkdir(cls.image_folder)
        os.mkdir(cls.ann_folder)

        cls.fake_img_paths = [f'{i}' for i in range(6)]
        cls.fake_labels = [[
            np.random.randint(10) for _ in range(np.random.randint(1, 4))
        ] for _ in range(6)]
        cls.fake_classes = [f'C_{i}' for i in range(10)]
        train_list = [i for i in range(0, 4)]
        test_list = [i for i in range(4, 6)]

        with open(osp.join(cls.image_set_folder, 'trainval.txt'), 'w') as f:
            for train_item in train_list:
                f.write(str(train_item) + '\n')
        with open(osp.join(cls.image_set_folder, 'test.txt'), 'w') as f:
            for test_item in test_list:
                f.write(str(test_item) + '\n')
        with open(osp.join(cls.image_set_folder, 'full_path_test.txt'),
                  'w') as f:
            for test_item in test_list:
                f.write(osp.join(cls.image_folder, str(test_item)) + '\n')

        for train_item in train_list:
            with open(osp.join(cls.ann_folder, f'{train_item}.xml'), 'w') as f:
                temple = '<object><name>C_{}</name>{}</object>'
                ann_data = ''.join([
                    temple.format(label, '<difficult>0</difficult>')
                    for label in cls.fake_labels[train_item]
                ])
                # add difficult label
                ann_data += ''.join([
                    temple.format(label, '<difficult>1</difficult>')
                    for label in cls.fake_labels[train_item]
                ])
                xml_ann_data = f'<annotation>{ann_data}</annotation>'
                f.write(xml_ann_data + '\n')

        for test_item in test_list:
            with open(osp.join(cls.ann_folder, f'{test_item}.xml'), 'w') as f:
                temple = '<object><name>C_{}</name>{}</object>'
                ann_data = ''.join([
                    temple.format(label, '<difficult>0</difficult>')
                    for label in cls.fake_labels[test_item]
                ])
                xml_ann_data = f'<annotation>{ann_data}</annotation>'
                f.write(xml_ann_data + '\n')

    def test_initialize(self):
        dataset_class = DATASETS.get(self.DATASET_TYPE)

        # Test overriding metainfo by `classes` argument
        cfg = {**self.DEFAULT_ARGS, 'classes': ['bus', 'car']}
        dataset = dataset_class(**cfg)
        self.assertEqual(dataset.CLASSES, ('bus', 'car'))

        # Test overriding CLASSES by classes file
        classes_file = osp.join(ASSETS_ROOT, 'classes.txt')
        cfg = {**self.DEFAULT_ARGS, 'classes': classes_file}
        dataset = dataset_class(**cfg)
        self.assertEqual(dataset.CLASSES, ('bus', 'car'))
        self.assertEqual(dataset.class_to_idx, {'bus': 0, 'car': 1})

        # Test invalid classes
        cfg = {**self.DEFAULT_ARGS, 'classes': dict(classes=1)}
        with self.assertRaisesRegex(ValueError, "type <class 'dict'>"):
            dataset_class(**cfg)

        # Test invalid split
        with self.assertRaisesRegex(AssertionError, 'The split must be'):
            cfg = {**self.DEFAULT_ARGS}
            cfg['split'] = 'unknown'
            dataset_class(**cfg)

        # Test valid splits
        splits = ['trainval', 'test']
        for split in splits:
            cfg = {**self.DEFAULT_ARGS}
            cfg['split'] = split
            dataset = dataset_class(**cfg)
            self.assertEqual(dataset.split, split)

        # Test split='trainval' and test_mode = True
        logger = MMLogger.get_current_instance()
        with self.assertLogs(logger, 'WARN') as log:
            cfg = {**self.DEFAULT_ARGS}
            cfg['split'] = 'trainval'
            cfg['test_mode'] = True
            dataset = dataset_class(**cfg)
            self.assertEqual(dataset.split, 'trainval')
            self.assertEqual(dataset.test_mode, True)
        self.assertIn('The trainval set will be used', log.output[0])

    def test_get_cat_ids(self):
        dataset_class = DATASETS.get(self.DATASET_TYPE)
        cfg = {'classes': self.fake_classes, **self.DEFAULT_ARGS}
        dataset = dataset_class(**cfg)

        cat_ids = dataset.get_cat_ids(0)
        self.assertIsInstance(cat_ids, list)
        self.assertIsInstance(cat_ids[0], int)

    def test_load_data_list(self):
        dataset_class = DATASETS.get(self.DATASET_TYPE)

        # Test default behavior
        dataset = dataset_class(**self.DEFAULT_ARGS)
        self.assertEqual(len(dataset), 4)
        self.assertEqual(len(dataset.CLASSES), 20)

        cfg = {
            'classes': self.fake_classes,
            'lazy_init': True,
            **self.DEFAULT_ARGS
        }
        dataset = dataset_class(**cfg)

        self.assertIn('Haven\'t been initialized', repr(dataset))
        dataset.full_init()
        self.assertIn(f'Number of samples: \t{len(dataset)}', repr(dataset))

        data_info = dataset[0]
        fake_img_path = osp.join(self.image_folder, self.fake_img_paths[0])
        self.assertEqual(data_info['img_path'], f'{fake_img_path}.jpg')
        self.assertEqual(set(data_info['gt_label']), set(self.fake_labels[0]))

        # Test with split='test'
        cfg['split'] = 'test'
        dataset = dataset_class(**cfg)
        self.assertEqual(len(dataset), 2)

        data_info = dataset[0]
        fake_img_path = osp.join(self.image_folder, self.fake_img_paths[4])
        self.assertEqual(data_info['img_path'], f'{fake_img_path}.jpg')
        self.assertEqual(set(data_info['gt_label']), set(self.fake_labels[4]))

        # Test with test_mode=True and ann_path = None
        cfg['split'] = ''
        cfg['image_set_path'] = 'ImageSets/Main/test.txt'
        cfg['test_mode'] = True
        cfg['data_prefix'] = 'JPEGImages'
        dataset = dataset_class(**cfg)
        self.assertEqual(len(dataset), 2)

        data_info = dataset[0]
        fake_img_path = osp.join(self.image_folder, self.fake_img_paths[4])
        self.assertEqual(data_info['img_path'], f'{fake_img_path}.jpg')
        self.assertEqual(data_info['gt_label'], None)

        # Test different backend
        cfg = {
            **self.DEFAULT_ARGS, 'lazy_init': True,
            'data_root': 's3://openmmlab/voc'
        }
        petrel_mock = MagicMock()
        sys.modules['petrel_client'] = petrel_mock
        dataset = dataset_class(**cfg)
        petrel_mock.client.Client.assert_called()

    def test_extra_repr(self):
        dataset_class = DATASETS.get(self.DATASET_TYPE)
        cfg = {**self.DEFAULT_ARGS}
        dataset = dataset_class(**cfg)

        self.assertIn(f'Path of image set: \t{dataset.image_set_path}',
                      repr(dataset))
        self.assertIn(f'Prefix of dataset: \t{dataset.data_root}',
                      repr(dataset))
        self.assertIn(f'Prefix of annotations: \t{dataset.ann_prefix}',
                      repr(dataset))
        self.assertIn(f'Prefix of images: \t{dataset.img_prefix}',
                      repr(dataset))

    @classmethod
    def tearDownClass(cls):
        cls.tmpdir.cleanup()


class TestMNIST(TestBaseDataset):
    DATASET_TYPE = 'MNIST'

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()

        tmpdir = tempfile.TemporaryDirectory()
        cls.tmpdir = tmpdir
        cls.root = tmpdir.name
        data_prefix = tmpdir.name
        cls.DEFAULT_ARGS = dict(data_root=cls.root, split='train')

        dataset_class = DATASETS.get(cls.DATASET_TYPE)

        def rm_suffix(s):
            return s[:s.rfind('.')]

        train_image_file = osp.join(data_prefix,
                                    rm_suffix(dataset_class.train_list[0][0]))
        train_label_file = osp.join(data_prefix,
                                    rm_suffix(dataset_class.train_list[1][0]))
        test_image_file = osp.join(data_prefix,
                                   rm_suffix(dataset_class.test_list[0][0]))
        test_label_file = osp.join(data_prefix,
                                   rm_suffix(dataset_class.test_list[1][0]))
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

    def test_initialize(self):
        dataset_class = DATASETS.get(self.DATASET_TYPE)

        # Test with invalid split
        with self.assertRaisesRegex(AssertionError, 'The split must be'):
            cfg = {**self.DEFAULT_ARGS}
            cfg['split'] = 'unknown'
            dataset_class(**cfg)

        # Test with valid split
        splits = ['train', 'test']
        test_modes = [False, True]

        for split in splits:
            for test_mode in test_modes:
                cfg = {**self.DEFAULT_ARGS}
                cfg['split'] = split
                cfg['test_mode'] = test_mode

                if split == 'train' and test_mode:
                    logger = MMLogger.get_current_instance()
                    with self.assertLogs(logger, 'WARN') as log:
                        dataset = dataset_class(**cfg)
                        self.assertEqual(dataset.split, split)
                        self.assertEqual(dataset.test_mode, test_mode)
                        self.assertEqual(dataset.data_root, self.root)
                    self.assertIn('training set will be used', log.output[0])
                else:
                    dataset = dataset_class(**cfg)
                    self.assertEqual(dataset.split, split)
                    self.assertEqual(dataset.test_mode, test_mode)
                    self.assertEqual(dataset.data_root, self.root)

    def test_load_data_list(self):
        dataset_class = DATASETS.get(self.DATASET_TYPE)

        # Test default behavior
        dataset = dataset_class(**self.DEFAULT_ARGS)
        self.assertEqual(len(dataset), 1)
        self.assertEqual(dataset.CLASSES, dataset_class.METAINFO['classes'])

        data_info = dataset[0]
        np.testing.assert_equal(data_info['img'], self.fake_img)
        np.testing.assert_equal(data_info['gt_label'], self.fake_label)

        # Test with split='test'
        cfg = {**self.DEFAULT_ARGS, 'split': 'test'}
        dataset = dataset_class(**cfg)
        self.assertEqual(len(dataset), 1)

        data_info = dataset[0]
        np.testing.assert_equal(data_info['img'], self.fake_img)
        np.testing.assert_equal(data_info['gt_label'], self.fake_label)

        # Test automatically download
        with patch('mmpretrain.datasets.mnist.download_and_extract_archive'
                   ) as mock:
            cfg = {**self.DEFAULT_ARGS, 'lazy_init': True, 'split': 'test'}
            dataset = dataset_class(**cfg)
            dataset.train_list = [['invalid_train_file', None]]
            dataset.test_list = [['invalid_test_file', None]]
            with self.assertRaisesRegex(AssertionError, 'Download failed'):
                dataset.full_init()
            calls = [
                call(
                    osp.join(dataset.url_prefix, dataset.train_list[0][0]),
                    download_root=dataset.data_prefix['root'],
                    filename=dataset.train_list[0][0],
                    md5=None),
                call(
                    osp.join(dataset.url_prefix, dataset.test_list[0][0]),
                    download_root=dataset.data_prefix['root'],
                    filename=dataset.test_list[0][0],
                    md5=None)
            ]
            mock.assert_has_calls(calls)

        with self.assertRaisesRegex(RuntimeError, '`download=True`'):
            cfg = {
                **self.DEFAULT_ARGS, 'lazy_init': True,
                'split': 'test',
                'download': False
            }
            dataset = dataset_class(**cfg)
            dataset._check_exists = MagicMock(return_value=False)
            dataset.full_init()

        # Test different backend
        cfg = {
            **self.DEFAULT_ARGS, 'lazy_init': True,
            'data_prefix': 'http://openmmlab/mnist'
        }
        dataset = dataset_class(**cfg)
        dataset._check_exists = MagicMock(return_value=False)
        with self.assertRaisesRegex(RuntimeError, 'http://openmmlab/mnist'):
            dataset.full_init()

    def test_extra_repr(self):
        dataset_class = DATASETS.get(self.DATASET_TYPE)
        cfg = {**self.DEFAULT_ARGS, 'lazy_init': True}
        dataset = dataset_class(**cfg)

        self.assertIn(f"Prefix of data: \t{dataset.data_prefix['root']}",
                      repr(dataset))

    @classmethod
    def tearDownClass(cls):
        cls.tmpdir.cleanup()


class FashionMNIST(TestMNIST):
    DATASET_TYPE = 'FashionMNIST'


class TestCUB(TestBaseDataset):
    DATASET_TYPE = 'CUB'

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()

        tmpdir = tempfile.TemporaryDirectory()
        cls.tmpdir = tmpdir
        cls.root = tmpdir.name
        cls.ann_file = 'images.txt'
        cls.image_folder = 'images'
        cls.image_class_labels_file = 'image_class_labels.txt'
        cls.train_test_split_file = 'train_test_split.txt'

        cls.DEFAULT_ARGS = dict(
            data_root=cls.root, split='train', test_mode=False)

        with open(osp.join(cls.root, cls.ann_file), 'w') as f:
            f.write('\n'.join([
                '1 1.txt',
                '2 2.txt',
                '3 3.txt',
            ]))

        with open(osp.join(cls.root, cls.image_class_labels_file), 'w') as f:
            f.write('\n'.join([
                '1 2',
                '2 3',
                '3 1',
            ]))

        with open(osp.join(cls.root, cls.train_test_split_file), 'w') as f:
            f.write('\n'.join([
                '1 0',
                '2 1',
                '3 1',
            ]))

    def test_initialize(self):
        dataset_class = DATASETS.get(self.DATASET_TYPE)

        # Test with invalid split
        with self.assertRaisesRegex(AssertionError, 'The split must be'):
            cfg = {**self.DEFAULT_ARGS}
            cfg['split'] = 'unknown'
            dataset_class(**cfg)

        # Test with valid split
        splits = ['train', 'test']
        test_modes = [False, True]

        for split in splits:
            for test_mode in test_modes:
                cfg = {**self.DEFAULT_ARGS}
                cfg['split'] = split
                cfg['test_mode'] = test_mode

                if split == 'train' and test_mode:
                    logger = MMLogger.get_current_instance()
                    with self.assertLogs(logger, 'WARN') as log:
                        dataset = dataset = dataset_class(**cfg)
                        self.assertEqual(dataset.split, split)
                        self.assertEqual(dataset.test_mode, test_mode)
                        self.assertEqual(dataset.data_root, self.root)
                        self.assertEqual(dataset.ann_file,
                                         osp.join(self.root, self.ann_file))
                    self.assertIn('training set will be used', log.output[0])
                else:
                    dataset = dataset_class(**cfg)
                    self.assertEqual(dataset.split, split)
                    self.assertEqual(dataset.test_mode, test_mode)
                    self.assertEqual(dataset.data_root, self.root)
                    self.assertEqual(dataset.ann_file,
                                     osp.join(self.root, self.ann_file))

    def test_load_data_list(self):
        dataset_class = DATASETS.get(self.DATASET_TYPE)

        # Test default behavior
        dataset = dataset_class(**self.DEFAULT_ARGS)
        self.assertEqual(len(dataset), 2)

        data_info = dataset[0]
        self.assertEqual(data_info['img_path'],
                         osp.join(self.root, self.image_folder, '2.txt'))
        self.assertEqual(data_info['gt_label'], 3 - 1)

        # # Test with split='test'
        cfg = {**self.DEFAULT_ARGS, 'split': 'test'}
        dataset = dataset_class(**cfg)
        self.assertEqual(len(dataset), 1)

        data_info = dataset[0]
        self.assertEqual(data_info['img_path'],
                         osp.join(self.root, self.image_folder, '1.txt'))
        self.assertEqual(data_info['gt_label'], 2 - 1)

    def test_extra_repr(self):
        dataset_class = DATASETS.get(self.DATASET_TYPE)
        cfg = {**self.DEFAULT_ARGS}
        dataset = dataset_class(**cfg)

        self.assertIn(f'Root of dataset: \t{dataset.data_root}', repr(dataset))

    @classmethod
    def tearDownClass(cls):
        cls.tmpdir.cleanup()


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
        metainfo = {'tasks': ['gender', 'wear']}
        self.assertDictEqual(dataset.metainfo, metainfo)
        self.assertFalse(dataset.test_mode)

    def test_parse_data_info(self):
        dataset_class = DATASETS.get(self.DATASET_TYPE)
        dataset = dataset_class(**self.DEFAULT_ARGS)

        data = dataset.parse_data_info({
            'img_path': 'a.jpg',
            'gt_label': {
                'gender': 0
            }
        })
        self.assertDictContainsSubset(
            {
                'img_path': os.path.join(ASSETS_ROOT, 'a.jpg'),
                'gt_label': {
                    'gender': 0
                }
            }, data)
        np.testing.assert_equal(data['gt_label']['gender'], 0)

        # Test missing path
        with self.assertRaisesRegex(AssertionError, 'have `img_path` field'):
            dataset.parse_data_info(
                {'gt_label': {
                    'gender': 0,
                    'wear': [1, 0, 1, 0]
                }})

    def test_repr(self):
        dataset_class = DATASETS.get(self.DATASET_TYPE)
        dataset = dataset_class(**self.DEFAULT_ARGS)

        task_doc = ('For 2 tasks\n     gender \n     wear ')
        self.assertIn(task_doc, repr(dataset))

    def test_load_data_list(self):
        dataset_class = DATASETS.get(self.DATASET_TYPE)

        # Test default behavior
        dataset = dataset_class(**self.DEFAULT_ARGS)
        data = dataset.load_data_list(self.DEFAULT_ARGS['ann_file'])
        self.assertIsInstance(data, list)
        np.testing.assert_equal(len(data), 3)
        np.testing.assert_equal(data[0]['gt_label'], {'gender': 0})
        np.testing.assert_equal(data[1]['gt_label'], {
            'gender': 0,
            'wear': [1, 0, 1, 0]
        })


class TestInShop(TestBaseDataset):
    DATASET_TYPE = 'InShop'

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()

        tmpdir = tempfile.TemporaryDirectory()
        cls.tmpdir = tmpdir
        cls.root = tmpdir.name
        cls.list_eval_partition = 'Eval/list_eval_partition.txt'
        cls.DEFAULT_ARGS = dict(data_root=cls.root, split='train')
        cls.ann_file = osp.join(cls.root, cls.list_eval_partition)
        os.makedirs(osp.join(cls.root, 'Eval'))
        with open(cls.ann_file, 'w') as f:
            f.write('\n'.join([
                '8',
                'image_name item_id evaluation_status',
                f'{osp.join("img", "02_1_front.jpg")} id_00000002 train',
                f'{osp.join("img", "02_2_side.jpg")} id_00000002 train',
                f'{osp.join("img", "12_3_back.jpg")} id_00007982 gallery',
                f'{osp.join("img", "12_7_addition.jpg")} id_00007982 gallery',
                f'{osp.join("img", "13_1_front.jpg")} id_00007982 query',
                f'{osp.join("img", "13_2_side.jpg")} id_00007983 gallery',
                f'{osp.join("img", "13_3_back.jpg")} id_00007983 query ',
                f'{osp.join("img", "13_7_additional.jpg")} id_00007983 query',
            ]))

    def test_initialize(self):
        dataset_class = DATASETS.get(self.DATASET_TYPE)

        # Test with mode=train
        cfg = {**self.DEFAULT_ARGS}
        dataset = dataset_class(**cfg)
        self.assertEqual(dataset.split, 'train')
        self.assertEqual(dataset.data_root, self.root)
        self.assertEqual(dataset.ann_file, self.ann_file)

        # Test with mode=query
        cfg = {**self.DEFAULT_ARGS, 'split': 'query'}
        dataset = dataset_class(**cfg)
        self.assertEqual(dataset.split, 'query')
        self.assertEqual(dataset.data_root, self.root)
        self.assertEqual(dataset.ann_file, self.ann_file)

        # Test with mode=gallery
        cfg = {**self.DEFAULT_ARGS, 'split': 'gallery'}
        dataset = dataset_class(**cfg)
        self.assertEqual(dataset.split, 'gallery')
        self.assertEqual(dataset.data_root, self.root)
        self.assertEqual(dataset.ann_file, self.ann_file)

        # Test with mode=other
        cfg = {**self.DEFAULT_ARGS, 'split': 'other'}
        with self.assertRaisesRegex(AssertionError, "'split' of `InS"):
            dataset_class(**cfg)

    def test_load_data_list(self):
        dataset_class = DATASETS.get(self.DATASET_TYPE)

        # Test with mode=train
        cfg = {**self.DEFAULT_ARGS}
        dataset = dataset_class(**cfg)
        self.assertEqual(len(dataset), 2)
        data_info = dataset[0]
        self.assertEqual(
            data_info['img_path'],
            os.path.join(self.root, 'Img', 'img', '02_1_front.jpg'))
        self.assertEqual(data_info['gt_label'], 0)

        # Test with mode=query
        cfg = {**self.DEFAULT_ARGS, 'split': 'query'}
        dataset = dataset_class(**cfg)
        self.assertEqual(len(dataset), 3)
        data_info = dataset[0]
        self.assertEqual(
            data_info['img_path'],
            os.path.join(self.root, 'Img', 'img', '13_1_front.jpg'))
        self.assertEqual(data_info['gt_label'], [0, 1])

        # Test with mode=gallery
        cfg = {**self.DEFAULT_ARGS, 'split': 'gallery'}
        dataset = dataset_class(**cfg)
        self.assertEqual(len(dataset), 3)
        data_info = dataset[0]
        self.assertEqual(
            data_info['img_path'],
            os.path.join(self.root, 'Img', 'img', '12_3_back.jpg'))
        self.assertEqual(data_info['sample_idx'], 0)

    def test_extra_repr(self):
        dataset_class = DATASETS.get(self.DATASET_TYPE)
        cfg = {**self.DEFAULT_ARGS}
        dataset = dataset_class(**cfg)

        self.assertIn(f'Root of dataset: \t{dataset.data_root}', repr(dataset))

    @classmethod
    def tearDownClass(cls):
        cls.tmpdir.cleanup()


class TestFlowers102(TestBaseDataset):
    DATASET_TYPE = 'Flowers102'

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()

        tmpdir = tempfile.TemporaryDirectory()
        cls.tmpdir = tmpdir
        cls.root = tmpdir.name
        cls.DEFAULT_ARGS = dict(data_root=cls.root, split='train')
        cls.ann_file = osp.join(cls.root, 'imagelabels.mat')
        cls.train_test_split_file = osp.join(cls.root, 'setid.mat')

        mat4py.savemat(cls.ann_file,
                       {'labels': [1, 1, 2, 2, 2, 3, 3, 4, 4, 5]})
        mat4py.savemat(cls.train_test_split_file, {
            'trnid': [1, 3, 5],
            'valid': [7, 9],
            'tstid': [2, 4, 6, 8, 10],
        })

    def test_initialize(self):
        dataset_class = DATASETS.get(self.DATASET_TYPE)

        # Test invalid split
        with self.assertRaisesRegex(AssertionError, 'The split must be'):
            cfg = {**self.DEFAULT_ARGS}
            cfg['split'] = 'unknown'
            dataset_class(**cfg)

        # Test valid splits
        splits = ['train', 'val', 'trainval', 'test']
        for split in splits:
            cfg = {**self.DEFAULT_ARGS}
            cfg['split'] = split
            dataset = dataset_class(**cfg)
            self.assertEqual(dataset.split, split)
            self.assertEqual(dataset.data_root, self.root)
            self.assertEqual(dataset.ann_file, self.ann_file)

    def test_load_data_list(self):
        dataset_class = DATASETS.get(self.DATASET_TYPE)

        # Test with split="train"
        cfg = {**self.DEFAULT_ARGS}
        dataset = dataset_class(**cfg)
        self.assertEqual(len(dataset), 3)
        data_info = dataset[0]
        self.assertEqual(data_info['img_path'],
                         os.path.join(self.root, 'jpg', 'image_00001.jpg'))
        self.assertEqual(data_info['gt_label'], 0)

        # Test with split="val"
        cfg = {**self.DEFAULT_ARGS, 'split': 'val'}
        dataset = dataset_class(**cfg)
        self.assertEqual(len(dataset), 2)
        data_info = dataset[0]
        self.assertEqual(data_info['img_path'],
                         os.path.join(self.root, 'jpg', 'image_00007.jpg'))
        self.assertEqual(data_info['gt_label'], 2)

        # Test with split="trainval"
        cfg = {**self.DEFAULT_ARGS, 'split': 'trainval'}
        dataset = dataset_class(**cfg)
        self.assertEqual(len(dataset), 5)
        data_info = dataset[2]
        self.assertEqual(data_info['img_path'],
                         os.path.join(self.root, 'jpg', 'image_00005.jpg'))
        self.assertEqual(data_info['gt_label'], 1)

        # Test with split="test"
        cfg = {**self.DEFAULT_ARGS, 'split': 'test'}
        dataset = dataset_class(**cfg)
        self.assertEqual(len(dataset), 5)
        data_info = dataset[2]
        self.assertEqual(data_info['img_path'],
                         os.path.join(self.root, 'jpg', 'image_00006.jpg'))
        self.assertEqual(data_info['gt_label'], 2)

    def test_extra_repr(self):
        dataset_class = DATASETS.get(self.DATASET_TYPE)
        cfg = {**self.DEFAULT_ARGS}
        dataset = dataset_class(**cfg)

        self.assertIn(f'Root of dataset: \t{dataset.data_root}', repr(dataset))

    @classmethod
    def tearDownClass(cls):
        cls.tmpdir.cleanup()


class TestOxfordIIITPet(TestBaseDataset):
    DATASET_TYPE = 'OxfordIIITPet'

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()

        tmpdir = tempfile.TemporaryDirectory()
        cls.tmpdir = tmpdir
        cls.root = tmpdir.name
        cls.trainval_file = 'trainval.txt'
        cls.image_folder = 'images'
        cls.meta_folder = 'annotations'
        cls.test_file = 'test.txt'

        os.mkdir(osp.join(cls.root, cls.meta_folder))

        cls.DEFAULT_ARGS = dict(data_root=cls.root, split='trainval')

        with open(osp.join(cls.root, cls.meta_folder, cls.trainval_file),
                  'w') as f:
            f.write('\n'.join([
                'Abyssinian_100 1 1 1',
                'american_bulldog_100 2 2 1',
                'basset_hound_126 4 2 3',
            ]))

        with open(osp.join(cls.root, cls.meta_folder, cls.test_file),
                  'w') as f:
            f.write('\n'.join([
                'Abyssinian_204 1 1 1',
                'american_bulldog_208 2 2 1',
            ]))

    def test_initialize(self):
        dataset_class = DATASETS.get(self.DATASET_TYPE)

        # Test invalid split
        with self.assertRaisesRegex(AssertionError, 'The split must be'):
            cfg = {**self.DEFAULT_ARGS}
            cfg['split'] = 'unknown'
            dataset_class(**cfg)

        # Test valid splits
        splits = ['trainval', 'test']
        for split in splits:
            cfg = {**self.DEFAULT_ARGS}
            cfg['split'] = split
            dataset = dataset_class(**cfg)
            self.assertEqual(dataset.split, split)
            self.assertEqual(dataset.data_root, self.root)

    def test_load_data_list(self):
        dataset_class = DATASETS.get(self.DATASET_TYPE)

        # Test default behavior
        dataset = dataset_class(**self.DEFAULT_ARGS)
        self.assertEqual(len(dataset), 3)

        data_info = dataset[0]
        self.assertEqual(
            data_info['img_path'],
            osp.join(self.root, self.image_folder, 'Abyssinian_100.jpg'))
        self.assertEqual(data_info['gt_label'], 1 - 1)

        # Test with split="test"
        cfg = {**self.DEFAULT_ARGS, 'split': 'test'}
        dataset = dataset_class(**cfg)
        self.assertEqual(len(dataset), 2)

        data_info = dataset[0]
        self.assertEqual(
            data_info['img_path'],
            osp.join(self.root, self.image_folder, 'Abyssinian_204.jpg'))
        self.assertEqual(data_info['gt_label'], 1 - 1)

    def test_extra_repr(self):
        dataset_class = DATASETS.get(self.DATASET_TYPE)
        cfg = {**self.DEFAULT_ARGS}
        dataset = dataset_class(**cfg)

        self.assertIn(f'Root of dataset: \t{dataset.data_root}', repr(dataset))

    @classmethod
    def tearDownClass(cls):
        cls.tmpdir.cleanup()


class TestDTD(TestBaseDataset):
    DATASET_TYPE = 'DTD'

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()

        tmpdir = tempfile.TemporaryDirectory()
        cls.tmpdir = tmpdir
        cls.root = tmpdir.name
        cls.DEFAULT_ARGS = dict(data_root=cls.root, split='train')

        cls.meta_folder = 'imdb'

        os.makedirs(osp.join(cls.root, cls.meta_folder))

        cls.ann_file = osp.join(cls.root, cls.meta_folder, 'imdb.mat')

        mat4py.savemat(
            cls.ann_file, {
                'images': {
                    'name': [
                        '1.jpg', '2.jpg', '3.jpg', '4.jpg', '5.jpg', '6.jpg',
                        '7.jpg', '8.jpg', '9.jpg', '10.jpg'
                    ],
                    'class': [1, 1, 2, 2, 2, 3, 3, 4, 4, 5],
                    'set': [1, 2, 3, 1, 2, 3, 1, 2, 3, 1]
                }
            })

    def test_initialize(self):
        dataset_class = DATASETS.get(self.DATASET_TYPE)

        # Test invalid split
        with self.assertRaisesRegex(AssertionError, 'The split must be'):
            cfg = {**self.DEFAULT_ARGS}
            cfg['split'] = 'unknown'
            dataset_class(**cfg)

        # Test valid splits
        splits = ['train', 'val', 'trainval', 'test']
        for split in splits:
            cfg = {**self.DEFAULT_ARGS}
            cfg['split'] = split
            dataset = dataset_class(**cfg)
            self.assertEqual(dataset.split, split)
            self.assertEqual(dataset.data_root, self.root)
            self.assertEqual(dataset.ann_file, self.ann_file)

    def test_load_data_list(self):
        dataset_class = DATASETS.get(self.DATASET_TYPE)

        # Test with split="train"
        cfg = {**self.DEFAULT_ARGS}
        dataset = dataset_class(**cfg)
        self.assertEqual(len(dataset), 4)
        data_info = dataset[0]
        self.assertEqual(data_info['img_path'],
                         os.path.join(self.root, 'images', '1.jpg'))
        self.assertEqual(data_info['gt_label'], 0)

        # Test with split="val"
        cfg = {**self.DEFAULT_ARGS, 'split': 'val'}
        dataset = dataset_class(**cfg)
        self.assertEqual(len(dataset), 3)
        data_info = dataset[0]
        self.assertEqual(data_info['img_path'],
                         os.path.join(self.root, 'images', '2.jpg'))
        self.assertEqual(data_info['gt_label'], 0)

        # Test with split="trainval"
        cfg = {**self.DEFAULT_ARGS, 'split': 'trainval'}
        dataset = dataset_class(**cfg)
        self.assertEqual(len(dataset), 7)
        data_info = dataset[2]
        self.assertEqual(data_info['img_path'],
                         os.path.join(self.root, 'images', '4.jpg'))
        self.assertEqual(data_info['gt_label'], 1)

        # Test with split="test"
        cfg = {**self.DEFAULT_ARGS, 'split': 'test'}
        dataset = dataset_class(**cfg)
        self.assertEqual(len(dataset), 3)
        data_info = dataset[0]
        self.assertEqual(data_info['img_path'],
                         os.path.join(self.root, 'images', '3.jpg'))
        self.assertEqual(data_info['gt_label'], 1)

    def test_extra_repr(self):
        dataset_class = DATASETS.get(self.DATASET_TYPE)
        cfg = {**self.DEFAULT_ARGS}
        dataset = dataset_class(**cfg)

        self.assertIn(f'Root of dataset: \t{dataset.data_root}', repr(dataset))

    @classmethod
    def tearDownClass(cls):
        cls.tmpdir.cleanup()


class TestFGVCAircraft(TestBaseDataset):
    DATASET_TYPE = 'FGVCAircraft'

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()

        tmpdir = tempfile.TemporaryDirectory()
        cls.tmpdir = tmpdir
        cls.root = tmpdir.name

        os.makedirs(osp.join(cls.root, 'data'))

        cls.train_file = osp.join('data', 'images_variant_train.txt')
        cls.val_file = osp.join('data', 'images_variant_val.txt')
        cls.trainval_file = osp.join('data', 'images_variant_trainval.txt')
        cls.test_file = osp.join('data', 'images_variant_test.txt')
        cls.image_folder = osp.join('data', 'images')

        cls.DEFAULT_ARGS = dict(data_root=cls.root, split='trainval')

        with open(osp.join(cls.root, cls.train_file), 'w') as f:
            f.write('\n'.join([
                '1025794 707-320',
                '1019011 727-200',
            ]))

        with open(osp.join(cls.root, cls.val_file), 'w') as f:
            f.write('\n'.join([
                '0209554 737-200',
            ]))

        with open(osp.join(cls.root, cls.trainval_file), 'w') as f:
            f.write('\n'.join([
                '1025794 707-320',
                '1019011 727-200',
                '0209554 737-200',
            ]))

        with open(osp.join(cls.root, cls.test_file), 'w') as f:
            f.write('\n'.join([
                '1514522 707-320',
                '0116175 727-200',
                '0713752 737-200',
                '2126017 737-300',
            ]))

    def test_initialize(self):
        dataset_class = DATASETS.get(self.DATASET_TYPE)

        # Test invalid split
        with self.assertRaisesRegex(AssertionError, 'The split must be'):
            cfg = {**self.DEFAULT_ARGS}
            cfg['split'] = 'unknown'
            dataset_class(**cfg)

        # Test valid splits
        splits = ['train', 'val', 'trainval', 'test']
        ann_files = [
            self.train_file, self.val_file, self.trainval_file, self.test_file
        ]
        for i, split in enumerate(splits):
            cfg = {**self.DEFAULT_ARGS}
            cfg['split'] = split
            dataset = dataset_class(**cfg)
            self.assertEqual(dataset.split, split)
            self.assertEqual(dataset.data_root, self.root)
            self.assertEqual(dataset.ann_file,
                             osp.join(self.root, ann_files[i]))

    def test_load_data_list(self):
        dataset_class = DATASETS.get(self.DATASET_TYPE)

        # Test default behavior (split="trainval")
        dataset = dataset_class(**self.DEFAULT_ARGS)
        self.assertEqual(len(dataset), 3)

        data_info = dataset[0]
        self.assertEqual(data_info['img_path'],
                         osp.join(self.root, self.image_folder, '1025794.jpg'))
        self.assertEqual(data_info['gt_label'], 0)

        # # Test with split="train"
        cfg = {**self.DEFAULT_ARGS, 'split': 'train'}
        dataset = dataset_class(**cfg)
        self.assertEqual(len(dataset), 2)

        data_info = dataset[0]
        self.assertEqual(data_info['img_path'],
                         osp.join(self.root, self.image_folder, '1025794.jpg'))
        self.assertEqual(data_info['gt_label'], 0)

        # Test with split="val"
        cfg = {**self.DEFAULT_ARGS, 'split': 'val'}
        dataset = dataset_class(**cfg)
        self.assertEqual(len(dataset), 1)

        data_info = dataset[0]
        self.assertEqual(data_info['img_path'],
                         osp.join(self.root, self.image_folder, '0209554.jpg'))
        self.assertEqual(data_info['gt_label'], 2)

        # Test with split="test"
        cfg = {**self.DEFAULT_ARGS, 'split': 'test'}
        dataset = dataset_class(**cfg)
        self.assertEqual(len(dataset), 4)

        data_info = dataset[0]
        self.assertEqual(data_info['img_path'],
                         osp.join(self.root, self.image_folder, '1514522.jpg'))
        self.assertEqual(data_info['gt_label'], 0)

    def test_extra_repr(self):
        dataset_class = DATASETS.get(self.DATASET_TYPE)
        cfg = {**self.DEFAULT_ARGS}
        dataset = dataset_class(**cfg)

        self.assertIn(f'Root of dataset: \t{dataset.data_root}', repr(dataset))

    @classmethod
    def tearDownClass(cls):
        cls.tmpdir.cleanup()


class TestStanfordCars(TestBaseDataset):
    DATASET_TYPE = 'StanfordCars'

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()

        tmpdir = tempfile.TemporaryDirectory()
        cls.tmpdir = tmpdir
        cls.root = tmpdir.name
        cls.ann_file = osp.join(cls.root, 'cars_annos.mat')
        cls.meta_folder = 'devkit'
        cls.train_ann_file = osp.join(cls.root, cls.meta_folder,
                                      'cars_train_annos.mat')
        cls.test_ann_file = osp.join(cls.root, cls.meta_folder,
                                     'cars_test_annos_withlabels.mat')
        cls.train_folder = 'cars_train'
        cls.test_folder = 'cars_test'

        os.makedirs(osp.join(cls.root, cls.meta_folder))

        cls.DEFAULT_ARGS = dict(data_root=cls.root, split='train')

        mat4py.savemat(
            cls.ann_file, {
                'annotations': {
                    'relative_im_path':
                    ['car_ims/001.jpg', 'car_ims/002.jpg', 'car_ims/003.jpg'],
                    'class': [1, 2, 3],
                    'test': [0, 0, 1]
                }
            })

        mat4py.savemat(
            cls.train_ann_file, {
                'annotations': {
                    'fname': ['001.jpg', '002.jpg', '012.jpg'],
                    'class': [10, 15, 150],
                }
            })

        mat4py.savemat(
            cls.test_ann_file, {
                'annotations': {
                    'fname': ['025.jpg', '111.jpg', '222.jpg'],
                    'class': [150, 1, 15],
                }
            })

    def test_initialize(self):
        dataset_class = DATASETS.get(self.DATASET_TYPE)

        # Test first way
        # Test invalid split
        with self.assertRaisesRegex(AssertionError, 'The split must be'):
            cfg = {**self.DEFAULT_ARGS}
            cfg['split'] = 'unknown'
            dataset_class(**cfg)

        # Test valid splits
        splits = ['train', 'test']
        for split in splits:
            cfg = {**self.DEFAULT_ARGS}
            cfg['split'] = split
            dataset = dataset_class(**cfg)
            self.assertEqual(dataset.split, split)
            self.assertEqual(dataset.data_root, self.root)
            self.assertEqual(dataset.ann_file, self.ann_file)

        # Test second way
        os.rename(self.ann_file, self.ann_file + 'copy')
        # Test invalid split
        with self.assertRaisesRegex(AssertionError, 'The split must be'):
            cfg = {**self.DEFAULT_ARGS}
            cfg['split'] = 'unknown'
            dataset_class(**cfg)

        # Test valid splits
        cfg = {**self.DEFAULT_ARGS}
        cfg['split'] = 'train'
        dataset = dataset_class(**cfg)
        self.assertEqual(dataset.split, 'train')
        self.assertEqual(dataset.data_root, self.root)
        self.assertEqual(dataset.ann_file,
                         osp.join(self.meta_folder, self.train_ann_file))

        # Test valid splits
        cfg = {**self.DEFAULT_ARGS}
        cfg['split'] = 'test'
        dataset = dataset_class(**cfg)
        self.assertEqual(dataset.split, 'test')
        self.assertEqual(dataset.data_root, self.root)
        self.assertEqual(dataset.ann_file,
                         osp.join(self.meta_folder, self.test_ann_file))

        # wrong dataset organization
        os.rename(self.train_ann_file, self.train_ann_file + 'copy')
        os.rename(self.test_ann_file, self.test_ann_file + 'copy')

        with self.assertRaisesRegex(RuntimeError,
                                    'The dataset is incorrectly organized'):
            cfg = {**self.DEFAULT_ARGS}
            dataset_class(**cfg)

        with self.assertRaisesRegex(RuntimeError,
                                    'The dataset is incorrectly organized'):
            cfg = {**self.DEFAULT_ARGS}
            cfg['split'] = 'test'
            dataset_class(**cfg)

        os.rename(self.train_ann_file + 'copy', self.train_ann_file)
        os.rename(self.test_ann_file + 'copy', self.test_ann_file)

        os.rename(self.ann_file + 'copy', self.ann_file)

    def test_load_data_list(self):
        dataset_class = DATASETS.get(self.DATASET_TYPE)

        # Test first way
        # Test default behavior
        assert osp.exists(osp.join(self.root, 'cars_annos.mat')), osp.join(
            self.root, 'cars_annos.mat')
        dataset = dataset_class(**self.DEFAULT_ARGS)
        self.assertEqual(len(dataset), 2)

        data_info = dataset[0]
        self.assertEqual(data_info['img_path'],
                         osp.join(self.root, 'car_ims/001.jpg'))
        self.assertEqual(data_info['gt_label'], 0)

        # Test with split="test"
        cfg = {**self.DEFAULT_ARGS, 'split': 'test'}
        dataset = dataset_class(**cfg)
        self.assertEqual(len(dataset), 1)

        data_info = dataset[0]
        self.assertEqual(data_info['img_path'],
                         osp.join(self.root, 'car_ims/003.jpg'))
        self.assertEqual(data_info['gt_label'], 2)

        # Test second way
        os.rename(self.ann_file, self.ann_file + 'copy')
        # Test with split="train"
        cfg = {**self.DEFAULT_ARGS}
        dataset = dataset_class(**cfg)
        self.assertEqual(len(dataset), 3)

        data_info = dataset[0]
        self.assertEqual(data_info['img_path'],
                         osp.join(self.root, self.train_folder, '001.jpg'))
        self.assertEqual(data_info['gt_label'], 9)

        # Test with split="test"
        cfg = {**self.DEFAULT_ARGS, 'split': 'test'}
        dataset = dataset_class(**cfg)
        self.assertEqual(len(dataset), 3)

        data_info = dataset[0]
        self.assertEqual(data_info['img_path'],
                         osp.join(self.root, self.test_folder, '025.jpg'))
        self.assertEqual(data_info['gt_label'], 149)

        os.rename(self.ann_file + 'copy', self.ann_file)

    def test_extra_repr(self):
        dataset_class = DATASETS.get(self.DATASET_TYPE)
        cfg = {**self.DEFAULT_ARGS}
        dataset = dataset_class(**cfg)

        self.assertIn(f'Root of dataset: \t{dataset.data_root}', repr(dataset))

    @classmethod
    def tearDownClass(cls):
        cls.tmpdir.cleanup()


class TestCaltech101(TestBaseDataset):
    DATASET_TYPE = 'Caltech101'

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()

        tmpdir = tempfile.TemporaryDirectory()
        cls.tmpdir = tmpdir
        cls.root = tmpdir.name
        cls.image_folder = '101_ObjectCategories'
        cls.meta_folder = 'meta'
        cls.train_file = 'train.txt'
        cls.test_file = 'test.txt'

        cls.DEFAULT_ARGS = dict(data_root=cls.root, split='train')

        os.makedirs(osp.join(cls.root, cls.meta_folder))

        with open(osp.join(cls.root, cls.meta_folder, cls.train_file),
                  'w') as f:
            f.write('\n'.join([
                '1.jpg 0',
                '2.jpg 1',
                '3.jpg 2',
            ]))

        with open(osp.join(cls.root, cls.meta_folder, cls.test_file),
                  'w') as f:
            f.write('\n'.join([
                '100.jpg 99',
                '101.jpg 100',
                '102.jpg 101',
                '103.jpg 101',
            ]))

    def test_initialize(self):
        dataset_class = DATASETS.get(self.DATASET_TYPE)

        # Test invalid split
        with self.assertRaisesRegex(AssertionError, 'The split must be'):
            cfg = {**self.DEFAULT_ARGS}
            cfg['split'] = 'unknown'
            dataset_class(**cfg)

        # Test valid splits
        splits = ['train', 'test']
        for split in splits:
            cfg = {**self.DEFAULT_ARGS}
            cfg['split'] = split
            dataset = dataset_class(**cfg)
            self.assertEqual(dataset.split, split)
            self.assertEqual(dataset.data_root, self.root)

    def test_load_data_list(self):
        dataset_class = DATASETS.get(self.DATASET_TYPE)

        # Test default behavior
        dataset = dataset_class(**self.DEFAULT_ARGS)
        self.assertEqual(len(dataset), 3)

        data_info = dataset[0]
        self.assertEqual(data_info['img_path'],
                         osp.join(self.root, self.image_folder, '1.jpg'))
        self.assertEqual(data_info['gt_label'], 0)

        # Test with split="test"
        cfg = {**self.DEFAULT_ARGS, 'split': 'test'}
        dataset = dataset_class(**cfg)
        self.assertEqual(len(dataset), 4)

        data_info = dataset[0]
        self.assertEqual(data_info['img_path'],
                         osp.join(self.root, self.image_folder, '100.jpg'))
        self.assertEqual(data_info['gt_label'], 99)

    def test_extra_repr(self):
        dataset_class = DATASETS.get(self.DATASET_TYPE)
        cfg = {**self.DEFAULT_ARGS}
        dataset = dataset_class(**cfg)

        self.assertIn(f'Root of dataset: \t{dataset.data_root}', repr(dataset))

    @classmethod
    def tearDownClass(cls):
        cls.tmpdir.cleanup()


class TestFood101(TestBaseDataset):
    DATASET_TYPE = 'Food101'

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()

        tmpdir = tempfile.TemporaryDirectory()
        cls.tmpdir = tmpdir
        cls.root = tmpdir.name
        cls.image_folder = 'images'
        cls.meta_folder = 'meta'
        cls.train_file = 'train.txt'
        cls.test_file = 'test.txt'

        os.makedirs(osp.join(cls.root, cls.meta_folder))

        cls.DEFAULT_ARGS = dict(data_root=cls.root, split='train')

        with open(osp.join(cls.root, cls.meta_folder, cls.train_file),
                  'w') as f:
            f.write('\n'.join([
                'apple_pie/0001',
                'baby_back_ribs/0002',
                'baklava/0003',
            ]))

        with open(osp.join(cls.root, cls.meta_folder, cls.test_file),
                  'w') as f:
            f.write('\n'.join([
                'beef_carpaccio/0004',
                'beef_tartare/0005',
                'beet_salad/0006',
            ]))

    def test_initialize(self):
        dataset_class = DATASETS.get(self.DATASET_TYPE)

        # Test invalid split
        with self.assertRaisesRegex(AssertionError, 'The split must be'):
            cfg = {**self.DEFAULT_ARGS}
            cfg['split'] = 'unknown'
            dataset_class(**cfg)

        # Test valid splits
        splits = ['train', 'test']
        for split in splits:
            cfg = {**self.DEFAULT_ARGS}
            cfg['split'] = split
            dataset = dataset_class(**cfg)
            self.assertEqual(dataset.split, split)
            self.assertEqual(dataset.data_root, self.root)

    def test_load_data_list(self):
        dataset_class = DATASETS.get(self.DATASET_TYPE)

        # Test default behavior
        dataset = dataset_class(**self.DEFAULT_ARGS)
        self.assertEqual(len(dataset), 3)

        data_info = dataset[0]
        self.assertEqual(
            data_info['img_path'],
            osp.join(self.root, self.image_folder, 'apple_pie', '0001.jpg'))
        self.assertEqual(data_info['gt_label'], 0)

        # Test with split="test"
        cfg = {**self.DEFAULT_ARGS, 'split': 'test'}
        dataset = dataset_class(**cfg)
        self.assertEqual(len(dataset), 3)

        data_info = dataset[0]
        self.assertEqual(
            data_info['img_path'],
            osp.join(self.root, self.image_folder, 'beef_carpaccio',
                     '0004.jpg'))
        self.assertEqual(data_info['gt_label'], 3)

    def test_extra_repr(self):
        dataset_class = DATASETS.get(self.DATASET_TYPE)
        cfg = {**self.DEFAULT_ARGS}
        dataset = dataset_class(**cfg)

        self.assertIn(f'Root of dataset: \t{dataset.data_root}', repr(dataset))

    @classmethod
    def tearDownClass(cls):
        cls.tmpdir.cleanup()


class TestSUN397(TestBaseDataset):
    DATASET_TYPE = 'SUN397'

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()

        tmpdir = tempfile.TemporaryDirectory()
        cls.tmpdir = tmpdir
        cls.root = tmpdir.name
        cls.train_file = 'Training_01.txt'
        cls.test_file = 'Testing_01.txt'
        cls.data_prefix = 'SUN397'
        cls.meta_folder = 'Partitions'

        os.makedirs(osp.join(cls.root, cls.meta_folder))

        cls.DEFAULT_ARGS = dict(data_root=cls.root, split='train')

        with open(osp.join(cls.root, cls.meta_folder, cls.train_file),
                  'w') as f:
            f.write('\n'.join([
                '/a/abbey/sun_aqswjsnjlrfzzhiz.jpg',
                '/a/airplane_cabin/sun_blczihbhbntqccux.jpg',
                '/a/assembly_line/sun_ajckcfldgdrdjogj.jpg',
            ]))

        with open(osp.join(cls.root, cls.meta_folder, cls.test_file),
                  'w') as f:
            f.write('\n'.join([
                '/a/abbey/sun_ajkqrqitspwywirx.jpg',
                '/a/airplane_cabin/sun_aqylhacwdsqfjuuu.jpg',
                '/a/auto_factory/sun_apfsprenzdnzbhmt.jpg',
                '/b/baggage_claim/sun_avittiqqaiibgcau.jpg',
            ]))

    def test_initialize(self):
        dataset_class = DATASETS.get(self.DATASET_TYPE)

        # Test invalid split
        with self.assertRaisesRegex(AssertionError, 'The split must be'):
            cfg = {**self.DEFAULT_ARGS}
            cfg['split'] = 'unknown'
            dataset_class(**cfg)

        # Test valid splits
        splits = ['train', 'test']
        for split in splits:
            cfg = {**self.DEFAULT_ARGS}
            cfg['split'] = split
            dataset = dataset_class(**cfg)
            self.assertEqual(dataset.split, split)
            self.assertEqual(dataset.data_root, self.root)

    def test_load_data_list(self):
        dataset_class = DATASETS.get(self.DATASET_TYPE)

        # Test default behavior
        dataset = dataset_class(**self.DEFAULT_ARGS)
        self.assertEqual(len(dataset), 3)
        data_info = dataset[0]
        self.assertEqual(
            data_info['img_path'],
            osp.join(self.root, self.data_prefix,
                     'a/abbey/sun_aqswjsnjlrfzzhiz.jpg'))
        self.assertEqual(data_info['gt_label'], 0)

        # Test with split='test'
        cfg = {**self.DEFAULT_ARGS, 'split': 'test'}
        dataset = dataset_class(**cfg)
        self.assertEqual(len(dataset), 4)
        data_info = dataset[-1]
        self.assertEqual(
            data_info['img_path'],
            osp.join(self.root, self.data_prefix,
                     'b/baggage_claim/sun_avittiqqaiibgcau.jpg'))
        self.assertEqual(data_info['gt_label'], 26)

    def test_extra_repr(self):
        dataset_class = DATASETS.get(self.DATASET_TYPE)
        cfg = {**self.DEFAULT_ARGS}
        dataset = dataset_class(**cfg)

        self.assertIn(f'Root of dataset: \t{dataset.data_root}', repr(dataset))

    @classmethod
    def tearDownClass(cls):
        cls.tmpdir.cleanup()
