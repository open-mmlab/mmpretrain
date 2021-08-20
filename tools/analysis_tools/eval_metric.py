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
        '--eval-options',
        nargs='+',
        action=DictAction,
        help='custom options for evaluation, the key-value pair in xxx=yyy '
        'format will be kwargs for dataset.evaluate() function')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)
    assert args.metrics, (
        'Please specify at least one metric the argument "--metrics".')

    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    # import modules from string list.
    if cfg.get('custom_imports', None):
        from mmcv.utils import import_modules_from_strings
        import_modules_from_strings(**cfg['custom_imports'])
    cfg.data.test.test_mode = True

    dataset = build_dataset(cfg.data.test)
    outputs = mmcv.load(args.pkl_results)
    pred_score = outputs['class_scores']

    kwargs = {} if args.eval_options is None else args.eval_options
    eval_kwargs = cfg.get('evaluation', {}).copy()
    # hard-code way to remove EvalHook args
    for key in [
            'interval', 'tmpdir', 'start', 'gpu_collect', 'save_best', 'rule'
    ]:
        eval_kwargs.pop(key, None)
    eval_kwargs.update(dict(metric=args.metrics, **kwargs))
    print(dataset.evaluate(pred_score, **eval_kwargs))


if __name__ == '__main__':
    main()
