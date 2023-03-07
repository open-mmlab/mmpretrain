# Copyright (c) OpenMMLab. All rights reserved.
import os
import os.path as osp
import pickle
import sys
import tempfile
from unittest import TestCase
from unittest.mock import MagicMock, call, patch

import numpy as np
from mmengine.logging import MMLogger
from scipy.io import matlab

from mmcls.registry import DATASETS, TRANSFORMS

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
        else:
            self.assertIn('The `CLASSES` meta info is not set.', repr(dataset))

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

    DEFAULT_ARGS = dict(data_root=ASSETS_ROOT, ann_file='ann.txt')

    def test_load_data_list(self):
        dataset_class = DATASETS.get(self.DATASET_TYPE)

        # test classes number
        cfg = {
            **self.DEFAULT_ARGS,
            'data_prefix': ASSETS_ROOT,
            'ann_file': '',
        }
        with self.assertRaisesRegex(
                AssertionError, r"\(2\) doesn't match .* classes \(1000\)"):
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


class TestImageNet21k(TestCustomDataset):
    DATASET_TYPE = 'ImageNet21k'

    DEFAULT_ARGS = dict(
        data_root=ASSETS_ROOT, classes=['cat', 'dog'], ann_file='ann.txt')

    def test_load_data_list(self):
        super().test_initialize()
        dataset_class = DATASETS.get(self.DATASET_TYPE)

        # The multi_label option is not implemented not.
        cfg = {**self.DEFAULT_ARGS, 'multi_label': True}
        with self.assertRaisesRegex(NotImplementedError, 'not supported'):
            dataset_class(**cfg)

        # Warn about ann_file
        cfg = {**self.DEFAULT_ARGS, 'ann_file': ''}
        logger = MMLogger.get_current_instance()
        with self.assertLogs(logger, 'WARN') as log:
            dataset_class(**cfg)
        self.assertIn('specify the `ann_file`', log.output[0])

        # Warn about classes
        cfg = {**self.DEFAULT_ARGS, 'classes': None}
        with self.assertLogs(logger, 'WARN') as log:
            dataset_class(**cfg)
        self.assertIn('specify the `classes`', log.output[0])


class TestCIFAR10(TestBaseDataset):
    DATASET_TYPE = 'CIFAR10'

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()

        tmpdir = tempfile.TemporaryDirectory()
        cls.tmpdir = tmpdir
        data_prefix = tmpdir.name
        cls.DEFAULT_ARGS = dict(
            data_prefix=data_prefix, pipeline=[], test_mode=False)

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

        # Test with test_mode=True
        cfg = {**self.DEFAULT_ARGS, 'test_mode': True}
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
        with patch(
                'mmcls.datasets.cifar.download_and_extract_archive') as mock:
            cfg = {**self.DEFAULT_ARGS, 'lazy_init': True, 'test_mode': True}
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
                'test_mode': True,
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

        cls.DEFAULT_ARGS = dict(
            data_root=data_root,
            image_set_path='ImageSets/train.txt',
            data_prefix=dict(img_path='JPEGImages', ann_path='Annotations'),
            pipeline=[],
            test_mode=False)

        cls.image_folder = osp.join(data_root, 'JPEGImages')
        cls.ann_folder = osp.join(data_root, 'Annotations')
        cls.image_set_folder = osp.join(data_root, 'ImageSets')
        os.mkdir(cls.image_set_folder)
        os.mkdir(cls.image_folder)
        os.mkdir(cls.ann_folder)

        cls.fake_img_paths = [f'{i}' for i in range(6)]
        cls.fake_labels = [[
            np.random.randint(10) for _ in range(np.random.randint(1, 4))
        ] for _ in range(6)]
        cls.fake_classes = [f'C_{i}' for i in range(10)]
        train_list = [i for i in range(0, 4)]
        test_list = [i for i in range(4, 6)]

        with open(osp.join(cls.image_set_folder, 'train.txt'), 'w') as f:
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

        # Test with test_mode=True
        cfg['image_set_path'] = 'ImageSets/test.txt'
        cfg['test_mode'] = True
        dataset = dataset_class(**cfg)
        self.assertEqual(len(dataset), 2)

        data_info = dataset[0]
        fake_img_path = osp.join(self.image_folder, self.fake_img_paths[4])
        self.assertEqual(data_info['img_path'], f'{fake_img_path}.jpg')
        self.assertEqual(set(data_info['gt_label']), set(self.fake_labels[4]))

        # Test with test_mode=True and ann_path = None
        cfg['image_set_path'] = 'ImageSets/test.txt'
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
        data_prefix = tmpdir.name
        cls.DEFAULT_ARGS = dict(
            data_prefix=data_prefix, pipeline=[], test_mode=False)

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

    def test_load_data_list(self):
        dataset_class = DATASETS.get(self.DATASET_TYPE)

        # Test default behavior
        dataset = dataset_class(**self.DEFAULT_ARGS)
        self.assertEqual(len(dataset), 1)
        self.assertEqual(dataset.CLASSES, dataset_class.METAINFO['classes'])

        data_info = dataset[0]
        np.testing.assert_equal(data_info['img'], self.fake_img)
        np.testing.assert_equal(data_info['gt_label'], self.fake_label)

        # Test with test_mode=True
        cfg = {**self.DEFAULT_ARGS, 'test_mode': True}
        dataset = dataset_class(**cfg)
        self.assertEqual(len(dataset), 1)

        data_info = dataset[0]
        np.testing.assert_equal(data_info['img'], self.fake_img)
        np.testing.assert_equal(data_info['gt_label'], self.fake_label)

        # Test automatically download
        with patch(
                'mmcls.datasets.mnist.download_and_extract_archive') as mock:
            cfg = {**self.DEFAULT_ARGS, 'lazy_init': True, 'test_mode': True}
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
                'test_mode': True,
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
        cls.ann_file = 'ann_file.txt'
        cls.image_folder = 'images'
        cls.image_class_labels_file = 'classes.txt'
        cls.train_test_split_file = 'split.txt'
        cls.train_test_split_file2 = 'split2.txt'
        cls.DEFAULT_ARGS = dict(
            data_root=cls.root,
            test_mode=False,
            data_prefix=cls.image_folder,
            pipeline=[],
            ann_file=cls.ann_file,
            image_class_labels_file=cls.image_class_labels_file,
            train_test_split_file=cls.train_test_split_file)

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

        with open(osp.join(cls.root, cls.train_test_split_file2), 'w') as f:
            f.write('\n'.join([
                '1 0',
                '2 1',
            ]))

    def test_load_data_list(self):
        dataset_class = DATASETS.get(self.DATASET_TYPE)

        # Test default behavior
        dataset = dataset_class(**self.DEFAULT_ARGS)
        self.assertEqual(len(dataset), 2)

        data_info = dataset[0]
        self.assertEqual(data_info['img_path'],
                         osp.join(self.root, self.image_folder, '2.txt'))
        self.assertEqual(data_info['gt_label'], 3 - 1)

        # Test with test_mode=True
        cfg = {**self.DEFAULT_ARGS, 'test_mode': True}
        dataset = dataset_class(**cfg)
        self.assertEqual(len(dataset), 1)

        data_info = dataset[0]
        self.assertEqual(data_info['img_path'],
                         osp.join(self.root, self.image_folder, '1.txt'))
        self.assertEqual(data_info['gt_label'], 2 - 1)

        # Test if the numbers of line are not match
        cfg = {
            **self.DEFAULT_ARGS, 'train_test_split_file':
            self.train_test_split_file2
        }
        with self.assertRaisesRegex(AssertionError,
                                    'sample_ids should be same'):
            dataset_class(**cfg)

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


class TestSVHN(TestBaseDataset):
    DATASET_TYPE = 'SVHN'

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()

        tmpdir = tempfile.TemporaryDirectory()
        cls.tmpdir = tmpdir
        data_prefix = tmpdir.name
        cls.DEFAULT_ARGS = dict(
            data_prefix=data_prefix, pipeline=[], test_mode=False)

        dataset_class = DATASETS.get(cls.DATASET_TYPE)

        train_file = osp.join(data_prefix, dataset_class.train_list[0][0])
        test_file = osp.join(data_prefix, dataset_class.test_list[0][0])
        cls.fake_img = np.ones((1, 3, 32, 32), dtype=np.uint8)
        cls.fake_label = np.zeros((1, ), dtype=np.uint8)

        for file in [train_file, test_file]:
            data = {
                'X': np.ones((32, 32, 3, 1), dtype=np.uint8),
                'y': np.ones((1, 1), dtype=np.uint8) * 10
            }
            matlab.savemat(file, data)

    def test_load_data_list(self):
        dataset_class = DATASETS.get(self.DATASET_TYPE)

        # Test default behavior
        dataset = dataset_class(**self.DEFAULT_ARGS)
        self.assertEqual(len(dataset), 1)
        self.assertEqual(dataset.CLASSES, dataset_class.METAINFO['classes'])

        data_info = dataset[0]
        np.testing.assert_equal(data_info['img'], self.fake_img)
        np.testing.assert_equal(data_info['gt_label'], self.fake_label)

        # Test with test_mode=True
        cfg = {**self.DEFAULT_ARGS, 'test_mode': True}
        dataset = dataset_class(**cfg)
        self.assertEqual(len(dataset), 1)

        data_info = dataset[0]
        np.testing.assert_equal(data_info['img'], self.fake_img)
        np.testing.assert_equal(data_info['gt_label'], self.fake_label)

        # Test automatically download
        with patch('mmcls.datasets.svhn.download_url') as mock:
            cfg = {**self.DEFAULT_ARGS, 'lazy_init': True, 'test_mode': True}
            dataset = dataset_class(**cfg)
            dataset.train_list = [['invalid_train_file', None]]
            dataset.test_list = [['invalid_test_file', None]]
            with self.assertRaisesRegex(AssertionError, 'Download failed'):
                dataset.full_init()
            calls = [
                call(
                    osp.join(dataset.url_prefix, dataset.train_list[0][0]),
                    dataset.data_prefix['root'],
                    filename=dataset.train_list[0][0],
                    md5=None),
                call(
                    osp.join(dataset.url_prefix, dataset.test_list[0][0]),
                    dataset.data_prefix['root'],
                    filename=dataset.test_list[0][0],
                    md5=None)
            ]
            mock.assert_has_calls(calls)

        with self.assertRaisesRegex(RuntimeError, '`download=True`'):
            cfg = {
                **self.DEFAULT_ARGS, 'lazy_init': True,
                'test_mode': True,
                'download': False
            }
            dataset = dataset_class(**cfg)
            dataset._check_exists = MagicMock(return_value=False)
            dataset.full_init()

        # Test different backend
        cfg = {
            **self.DEFAULT_ARGS, 'lazy_init': True,
            'data_prefix': 'http://openmmlab/svhn'
        }
        dataset = dataset_class(**cfg)
        dataset._check_exists = MagicMock(return_value=False)
        with self.assertRaisesRegex(RuntimeError, 'http://openmmlab/svhn'):
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
