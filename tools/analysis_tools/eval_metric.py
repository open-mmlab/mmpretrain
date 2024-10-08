# Copyright (c) OpenMMLab. All rights reserved.
import argparse

import mmcv
from mmcv import Config, DictAction

from mmcls.datasets import build_dataset


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate metric of the '
                                     'results saved in pkl format')
    parser.add_argument('config', help='Config of the model')
    parser.add_argument('pkl_results', help='Results in pickle format')
    parser.add_argument(
        '--metrics',
        type=str,
        nargs='+',
        help='Evaluation metrics, which depends on the dataset, e.g., '
        '"accuracy", "precision", "recall" and "support".')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--metric-options',
        nargs='+',
        action=DictAction,
        help='custom options for evaluation, the key-value pair in xxx=yyy '
        'format will be kwargs for dataset.evaluate() function')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    outputs = mmcv.load(args.pkl_results)
    assert 'class_scores' in outputs, \
        'No "class_scores" in result file, please set "--out-items" in test.py'

    cfg = Config.fromfile(args.config)
    assert args.metrics, (
        'Please specify at least one metric the argument "--metrics".')

    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    cfg.data.test.test_mode = True

    dataset = build_dataset(cfg.data.test)
    pred_score = outputs['class_scores']

    eval_kwargs = cfg.get('evaluation', {}).copy()
    # hard-code way to remove EvalHook args
    for key in [
            'interval', 'tmpdir', 'start', 'gpu_collect', 'save_best', 'rule'
    ]:
        eval_kwargs.pop(key, None)
    eval_kwargs.update(
        dict(metric=args.metrics, metric_options=args.metric_options))
    print(dataset.evaluate(pred_score, **eval_kwargs))


if __name__ == '__main__':
    main()
