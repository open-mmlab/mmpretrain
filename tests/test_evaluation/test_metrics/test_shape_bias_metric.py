# Copyright (c) OpenMMLab. All rights reserved.
import torch

from mmpretrain.evaluation import ShapeBiasMetric


def test_shape_bias_metric():
    data_sample = dict()
    data_sample['pred_label'] = dict(
        score=torch.rand(1000, ), label=torch.tensor(1))
    data_sample['gt_label'] = dict(label=torch.tensor(1))
    data_sample['img_path'] = 'tests/airplane/test.JPEG'
    evaluator = ShapeBiasMetric(
        csv_dir='tests/data', dataset_name='test', model_name='test')
    evaluator.process(None, [data_sample])
