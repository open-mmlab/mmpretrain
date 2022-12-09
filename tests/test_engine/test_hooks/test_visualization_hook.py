# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import tempfile
from unittest import TestCase
from unittest.mock import ANY, MagicMock, patch

import torch
from mmengine.runner import EpochBasedTrainLoop, IterBasedTrainLoop

from mmcls.engine import VisualizationHook
from mmcls.registry import HOOKS
from mmcls.structures import ClsDataSample
from mmcls.utils import register_all_modules
from mmcls.visualization import ClsVisualizer

register_all_modules()


class TestVisualizationHook(TestCase):

    def setUp(self) -> None:
        ClsVisualizer.get_instance('visualizer')

        data_sample = ClsDataSample().set_gt_label(1).set_pred_label(2)
        data_sample.set_metainfo({'img_path': 'tests/data/color.jpg'})
        self.data_batch = {
            'inputs': torch.randint(0, 256, (10, 3, 224, 224)),
            'data_sample': [data_sample] * 10
        }

        self.outputs = [data_sample] * 10

        self.tmpdir = tempfile.TemporaryDirectory()

    def test_draw_samples(self):
        # test enable=False
        cfg = dict(type='VisualizationHook', enable=False)
        hook: VisualizationHook = HOOKS.build(cfg)
        with patch.object(hook._visualizer, 'add_datasample') as mock:
            hook._draw_samples(1, self.data_batch, self.outputs, step=1)
            mock.assert_not_called()

        # test enable=True
        cfg = dict(type='VisualizationHook', enable=True, show=True)
        hook: VisualizationHook = HOOKS.build(cfg)
        with patch.object(hook._visualizer, 'add_datasample') as mock:
            hook._draw_samples(0, self.data_batch, self.outputs, step=0)
            mock.assert_called_once_with(
                'color.jpg',
                image=ANY,
                data_sample=self.outputs[0],
                step=0,
                show=True)

        # test samples without path
        cfg = dict(type='VisualizationHook', enable=True)
        hook: VisualizationHook = HOOKS.build(cfg)
        with patch.object(hook._visualizer, 'add_datasample') as mock:
            outputs = [ClsDataSample()] * 10
            hook._draw_samples(0, self.data_batch, outputs, step=0)
            mock.assert_called_once_with(
                '0', image=ANY, data_sample=outputs[0], step=0, show=False)

        # test out_dir
        cfg = dict(
            type='VisualizationHook', enable=True, out_dir=self.tmpdir.name)
        hook: VisualizationHook = HOOKS.build(cfg)
        with patch.object(hook._visualizer, 'add_datasample') as mock:
            hook._draw_samples(0, self.data_batch, self.outputs, step=0)
            mock.assert_called_once_with(
                'color.jpg',
                image=ANY,
                data_sample=self.outputs[0],
                step=0,
                show=False,
                out_file=osp.join(self.tmpdir.name, 'color.jpg_0.png'))

        # test sample idx
        cfg = dict(type='VisualizationHook', enable=True, interval=4)
        hook: VisualizationHook = HOOKS.build(cfg)
        with patch.object(hook._visualizer, 'add_datasample') as mock:
            hook._draw_samples(1, self.data_batch, self.outputs, step=0)
            mock.assert_called_with(
                'color.jpg',
                image=ANY,
                data_sample=self.outputs[2],
                step=0,
                show=False)
            mock.assert_called_with(
                'color.jpg',
                image=ANY,
                data_sample=self.outputs[6],
                step=0,
                show=False)

    def test_after_val_iter(self):
        runner = MagicMock()

        # test epoch-based
        runner.train_loop = MagicMock(spec=EpochBasedTrainLoop)
        runner.epoch = 5
        cfg = dict(type='VisualizationHook', enable=True)
        hook = HOOKS.build(cfg)
        with patch.object(hook._visualizer, 'add_datasample') as mock:
            hook.after_val_iter(runner, 0, self.data_batch, self.outputs)
            mock.assert_called_once_with(
                'color.jpg',
                image=ANY,
                data_sample=self.outputs[0],
                step=5,
                show=False)

        # test iter-based
        runner.train_loop = MagicMock(spec=IterBasedTrainLoop)
        runner.iter = 300
        cfg = dict(type='VisualizationHook', enable=True)
        hook = HOOKS.build(cfg)
        with patch.object(hook._visualizer, 'add_datasample') as mock:
            hook.after_val_iter(runner, 0, self.data_batch, self.outputs)
            mock.assert_called_once_with(
                'color.jpg',
                image=ANY,
                data_sample=self.outputs[0],
                step=300,
                show=False)

    def test_after_test_iter(self):
        runner = MagicMock()

        cfg = dict(type='VisualizationHook', enable=True)
        hook = HOOKS.build(cfg)
        with patch.object(hook._visualizer, 'add_datasample') as mock:
            hook.after_test_iter(runner, 0, self.data_batch, self.outputs)
            mock.assert_called_once_with(
                'color.jpg',
                image=ANY,
                data_sample=self.outputs[0],
                step=0,
                show=False)

    def tearDown(self) -> None:
        self.tmpdir.cleanup()
