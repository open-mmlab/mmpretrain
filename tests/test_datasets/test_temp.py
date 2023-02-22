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

from mmpretrain.registry import DATASETS, TRANSFORMS

ASSETS_ROOT = osp.abspath(osp.join(osp.dirname(__file__), '../data/dataset'))


class TestCustomDataset(TestCase):
    DATASET_TYPE = 'CustomDataset'

    DEFAULT_ARGS = dict(data_root=ASSETS_ROOT, ann_file='ann.txt')

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

        # test load without ann_file and without labels (no folder structures)
        cfg = {
            **self.DEFAULT_ARGS,
            'data_prefix': ASSETS_ROOT,
            'ann_file': '',
            'with_label': False,
        }
        dataset = dataset_class(**cfg)
        self.assertEqual(len(dataset), 3)
        self.assertIsNone(dataset.CLASSES, None)
        self.assertGreaterEqual(
            dataset.get_data_info(0).items(), {
                'img_path': osp.join(ASSETS_ROOT, 'a', '1.JPG'),
            }.items())
        self.assertGreaterEqual(
            dataset.get_data_info(2).items(), {
                'img_path': osp.join(ASSETS_ROOT, 'b', 'subb', '3.jpg'),
            }.items())
