# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import tempfile

from mmpretrain.utils import load_json_log


def test_load_json_log():
    demo_log = """\
{"lr": 0.0001, "data_time": 0.003, "loss": 2.29, "time": 0.010, "epoch": 1, "step": 150}
{"lr": 0.0001, "data_time": 0.002, "loss": 2.28, "time": 0.007, "epoch": 1, "step": 300}
{"lr": 0.0001, "data_time": 0.001, "loss": 2.27, "time": 0.008, "epoch": 1, "step": 450}
{"accuracy/top1": 23.98, "accuracy/top5": 66.05, "step": 1}
{"lr": 0.0001, "data_time": 0.001, "loss": 2.25, "time": 0.014, "epoch": 2, "step": 619}
{"lr": 0.0001, "data_time": 0.000, "loss": 2.24, "time": 0.012, "epoch": 2, "step": 769}
{"lr": 0.0001, "data_time": 0.003, "loss": 2.23, "time": 0.009, "epoch": 2, "step": 919}
{"accuracy/top1": 41.82, "accuracy/top5": 81.26, "step": 2}
{"lr": 0.0001, "data_time": 0.002, "loss": 2.21, "time": 0.007, "epoch": 3, "step": 1088}
{"lr": 0.0001, "data_time": 0.005, "loss": 2.18, "time": 0.009, "epoch": 3, "step": 1238}
{"lr": 0.0001, "data_time": 0.002, "loss": 2.16, "time": 0.008, "epoch": 3, "step": 1388}
{"accuracy/top1": 54.07, "accuracy/top5": 89.80, "step": 3}
"""  # noqa: E501
    with tempfile.TemporaryDirectory() as tmpdir:
        json_log = osp.join(tmpdir, 'scalars.json')
        with open(json_log, 'w') as f:
            f.write(demo_log)

        log_dict = load_json_log(json_log)

    assert log_dict.keys() == {'train', 'val'}
    assert log_dict['train'][3] == {
        'lr': 0.0001,
        'data_time': 0.001,
        'loss': 2.25,
        'time': 0.014,
        'epoch': 2,
        'step': 619
    }
    assert log_dict['val'][2] == {
        'accuracy/top1': 54.07,
        'accuracy/top5': 89.80,
        'step': 3
    }
