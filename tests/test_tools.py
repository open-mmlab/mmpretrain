# Copyright (c) OpenMMLab. All rights reserved.
import re
import tempfile
from collections import OrderedDict
from pathlib import Path
from subprocess import PIPE, Popen
from unittest import TestCase

import mmengine
import torch
from mmengine.config import Config

from mmpretrain import ModelHub, get_model
from mmpretrain.structures import DataSample

MMPRE_ROOT = Path(__file__).parent.parent
ASSETS_ROOT = Path(__file__).parent / 'data'


class TestImageDemo(TestCase):

    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmpdir.name)

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_run(self):
        command = [
            'python',
            'demo/image_demo.py',
            'demo/demo.JPEG',
            'mobilevit-xxsmall_3rdparty_in1k',
            '--device',
            'cpu',
        ]
        p = Popen(command, cwd=MMPRE_ROOT, stdout=PIPE)
        out, _ = p.communicate()
        self.assertIn('sea snake', out.decode())


class TestAnalyzeLogs(TestCase):

    def setUp(self):
        self.log_file = ASSETS_ROOT / 'vis_data.json'
        self.tmpdir = tempfile.TemporaryDirectory()
        self.out_file = Path(self.tmpdir.name) / 'out.png'

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_run(self):
        command = [
            'python',
            'tools/analysis_tools/analyze_logs.py',
            'cal_train_time',
            str(self.log_file),
        ]
        p = Popen(command, cwd=MMPRE_ROOT, stdout=PIPE)
        out, _ = p.communicate()
        self.assertIn('slowest epoch 2, average time is 0.0219', out.decode())

        command = [
            'python',
            'tools/analysis_tools/analyze_logs.py',
            'plot_curve',
            str(self.log_file),
            '--keys',
            'accuracy/top1',
            '--out',
            str(self.out_file),
        ]
        p = Popen(command, cwd=MMPRE_ROOT, stdout=PIPE)
        out, _ = p.communicate()
        self.assertIn(str(self.log_file), out.decode())
        self.assertIn(str(self.out_file), out.decode())
        self.assertTrue(self.out_file.exists())


class TestAnalyzeResults(TestCase):

    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmpdir.name)

        dataset_cfg = dict(
            type='CustomDataset',
            data_root=str(ASSETS_ROOT / 'dataset'),
        )
        config = Config(dict(test_dataloader=dict(dataset=dataset_cfg)))
        self.config_file = self.dir / 'config.py'
        config.dump(self.config_file)

        results = [{
            'gt_label': 1,
            'pred_label': 0,
            'pred_score': [0.9, 0.1],
            'sample_idx': 0,
        }, {
            'gt_label': 0,
            'pred_label': 0,
            'pred_score': [0.9, 0.1],
            'sample_idx': 1,
        }]
        self.result_file = self.dir / 'results.pkl'
        mmengine.dump(results, self.result_file)

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_run(self):
        command = [
            'python',
            'tools/analysis_tools/analyze_results.py',
            str(self.config_file),
            str(self.result_file),
            '--out-dir',
            str(self.tmpdir.name),
        ]
        p = Popen(command, cwd=MMPRE_ROOT, stdout=PIPE)
        p.communicate()
        self.assertTrue((self.dir / 'success/2.jpeg.png').exists())
        self.assertTrue((self.dir / 'fail/1.JPG.png').exists())


class TestPrintConfig(TestCase):

    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.config_file = MMPRE_ROOT / 'configs/resnet/resnet18_8xb32_in1k.py'

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_run(self):
        command = [
            'python',
            'tools/misc/print_config.py',
            str(self.config_file),
        ]
        p = Popen(command, cwd=MMPRE_ROOT, stdout=PIPE)
        out, _ = p.communicate()
        out = out.decode().strip().replace('\r\n', '\n')
        self.assertEqual(out,
                         Config.fromfile(self.config_file).pretty_text.strip())


class TestVerifyDataset(TestCase):

    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmpdir.name)
        dataset_cfg = dict(
            type='CustomDataset',
            ann_file=str(self.dir / 'ann.txt'),
            pipeline=[dict(type='LoadImageFromFile')],
            data_root=str(ASSETS_ROOT / 'dataset'),
        )
        ann_file = '\n'.join(['a/2.JPG 0', 'b/2.jpeg 1', 'b/subb/3.jpg 1'])
        (self.dir / 'ann.txt').write_text(ann_file)
        config = Config(dict(train_dataloader=dict(dataset=dataset_cfg)))
        self.config_file = Path(self.tmpdir.name) / 'config.py'
        config.dump(self.config_file)

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_run(self):
        command = [
            'python',
            'tools/misc/verify_dataset.py',
            str(self.config_file),
            '--out-path',
            str(self.dir / 'log.log'),
        ]
        p = Popen(command, cwd=MMPRE_ROOT, stdout=PIPE)
        out, _ = p.communicate()
        self.assertIn(
            f"{ASSETS_ROOT/'dataset/a/2.JPG'} cannot be read correctly",
            out.decode().strip())
        self.assertTrue((self.dir / 'log.log').exists())


class TestEvalMetric(TestCase):

    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmpdir.name)

        results = [
            DataSample().set_gt_label(1).set_pred_label(0).set_pred_score(
                [0.6, 0.3, 0.1]).to_dict(),
            DataSample().set_gt_label(0).set_pred_label(0).set_pred_score(
                [0.6, 0.3, 0.1]).to_dict(),
        ]
        self.result_file = self.dir / 'results.pkl'
        mmengine.dump(results, self.result_file)

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_run(self):
        command = [
            'python',
            'tools/analysis_tools/eval_metric.py',
            str(self.result_file),
            '--metric',
            'type=Accuracy',
            'topk=1,2',
        ]
        p = Popen(command, cwd=MMPRE_ROOT, stdout=PIPE)
        out, _ = p.communicate()
        self.assertIn('accuracy/top1', out.decode())
        self.assertIn('accuracy/top2', out.decode())


class TestVisScheduler(TestCase):

    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmpdir.name)

        config = Config.fromfile(MMPRE_ROOT /
                                 'configs/resnet/resnet18_8xb32_in1k.py')
        config.param_scheduler = [
            dict(
                type='LinearLR',
                start_factor=0.01,
                by_epoch=True,
                end=1,
                convert_to_iter_based=True),
            dict(type='CosineAnnealingLR', by_epoch=True, begin=1),
        ]
        config.work_dir = str(self.dir)
        config.train_cfg.max_epochs = 2
        self.config_file = Path(self.tmpdir.name) / 'config.py'
        config.dump(self.config_file)

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_run(self):
        command = [
            'python',
            'tools/visualization/vis_scheduler.py',
            str(self.config_file),
            '--dataset-size',
            '100',
            '--not-show',
            '--save-path',
            str(self.dir / 'out.png'),
        ]
        p = Popen(command, cwd=MMPRE_ROOT, stdout=PIPE)
        p.communicate()
        self.assertTrue((self.dir / 'out.png').exists())


class TestPublishModel(TestCase):

    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmpdir.name)

        ckpt = dict(
            state_dict=OrderedDict({
                'a': torch.tensor(1.),
            }),
            ema_state_dict=OrderedDict({
                'step': 1,
                'module.a': torch.tensor(2.),
            }))
        self.ckpt_file = self.dir / 'ckpt.pth'
        torch.save(ckpt, self.ckpt_file)

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_run(self):
        command = [
            'python',
            'tools/model_converters/publish_model.py',
            str(self.ckpt_file),
            str(self.ckpt_file),
            '--dataset-type',
            'ImageNet',
            '--no-ema',
        ]
        p = Popen(command, cwd=MMPRE_ROOT, stdout=PIPE)
        out, _ = p.communicate()
        self.assertIn('and drop the EMA weights.', out.decode())
        self.assertIn('Successfully generated', out.decode())
        output_ckpt = re.findall(r'ckpt_\d{8}-\w{8}.pth', out.decode())
        self.assertGreater(len(output_ckpt), 0)
        output_ckpt = output_ckpt[0]
        self.assertTrue((self.dir / output_ckpt).exists())
        # The input file won't be overridden.
        self.assertTrue(self.ckpt_file.exists())


class TestVisCam(TestCase):

    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmpdir.name)

        model = get_model('mobilevit-xxsmall_3rdparty_in1k')
        self.config_file = self.dir / 'config.py'
        model._config.dump(self.config_file)

        self.ckpt_file = self.dir / 'ckpt.pth'
        torch.save(model.state_dict(), self.ckpt_file)

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_run(self):
        command = [
            'python',
            'tools/visualization/vis_cam.py',
            str(ASSETS_ROOT / 'color.jpg'),
            str(self.config_file),
            str(self.ckpt_file),
            '--save-path',
            str(self.dir / 'cam.jpg'),
        ]
        p = Popen(command, cwd=MMPRE_ROOT, stdout=PIPE)
        out, _ = p.communicate()
        self.assertIn('backbone.conv_1x1_exp.bn', out.decode())
        self.assertTrue((self.dir / 'cam.jpg').exists())


class TestConfusionMatrix(TestCase):

    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmpdir.name)

        self.config_file = MMPRE_ROOT / 'configs/resnet/resnet18_8xb32_in1k.py'

        results = [
            DataSample().set_gt_label(1).set_pred_label(0).set_pred_score(
                [0.6, 0.3, 0.1]).to_dict(),
            DataSample().set_gt_label(0).set_pred_label(0).set_pred_score(
                [0.6, 0.3, 0.1]).to_dict(),
        ]
        self.result_file = self.dir / 'results.pkl'
        mmengine.dump(results, self.result_file)

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_run(self):
        command = [
            'python',
            'tools/analysis_tools/confusion_matrix.py',
            str(self.config_file),
            str(self.result_file),
            '--out',
            str(self.dir / 'result.pkl'),
        ]
        Popen(command, cwd=MMPRE_ROOT, stdout=PIPE).wait()
        result = mmengine.load(self.dir / 'result.pkl')
        torch.testing.assert_allclose(
            result, torch.tensor([
                [1, 0, 0],
                [1, 0, 0],
                [0, 0, 0],
            ]))


class TestVisTsne(TestCase):

    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmpdir.name)

        config = ModelHub.get('mobilevit-xxsmall_3rdparty_in1k').config
        test_dataloader = dict(
            batch_size=1,
            dataset=dict(
                type='CustomDataset',
                data_root=str(ASSETS_ROOT / 'dataset'),
                pipeline=config.test_dataloader.dataset.pipeline,
            ),
            sampler=dict(type='DefaultSampler', shuffle=False),
        )
        config.test_dataloader = mmengine.ConfigDict(test_dataloader)
        self.config_file = self.dir / 'config.py'
        config.dump(self.config_file)

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_run(self):
        command = [
            'python',
            'tools/visualization/vis_tsne.py',
            str(self.config_file),
            '--work-dir',
            str(self.dir),
            '--perplexity',
            '2',
        ]
        Popen(command, cwd=MMPRE_ROOT, stdout=PIPE).wait()
        self.assertTrue(len(list(self.dir.glob('tsne_*/feat_*.png'))) > 0)


class TestGetFlops(TestCase):

    def test_run(self):
        command = [
            'python',
            'tools/analysis_tools/get_flops.py',
            'mobilevit-xxsmall_3rdparty_in1k',
        ]
        ret_code = Popen(command, cwd=MMPRE_ROOT).wait()
        self.assertEqual(ret_code, 0)
