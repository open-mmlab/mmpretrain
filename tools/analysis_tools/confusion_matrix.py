# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os.path as osp
import tempfile

import mmengine
from mmengine.logging import MMLogger
from mmengine.runner import Runner

from mmcls.evaluation import ConfusionMatrix
from mmcls.utils import register_all_modules


def parse_args():
    parser = argparse.ArgumentParser(
        description='Eval a checkpoint and draw the confusion matrix.')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--out', help='the file to save the confusion matrix.')
    parser.add_argument(
        '--show',
        action='store_true',
        help='whether to display the metric result by matplotlib if supports.')
    parser.add_argument(
        '--show-dir',
        help='directory where the visualization images will be saved.')
    args = parser.parse_args()
    return args


class SaveConfusionMatrix(ConfusionMatrix):

    def __init__(self, out, show, show_dir, **kwargs) -> None:
        self.out = out
        self.show = show
        self.show_dir = show_dir
        super().__init__(**kwargs)

    def compute_metrics(self, results: list) -> dict:
        metric = super().compute_metrics(results)

        confusion_matrix = metric['result']
        if self.dataset_meta is not None:
            classes = self.dataset_meta.get('classes', None)
        else:
            classes = None

        if self.out is not None:
            mmengine.dump(confusion_matrix, self.out)

        if self.show or self.show_dir is not None:
            fig = ConfusionMatrix.plot(
                confusion_matrix, show=self.show, classes=classes)
            if self.show_dir is not None:
                save_path = osp.join(self.show_dir, 'confusion_matrix.png')
                fig.savefig(save_path)
                MMLogger.get_current_instance().info(
                    f'The confusion matrix is saved at {save_path}.')
        return metric


def main():
    args = parse_args()

    # register all modules in mmcls into the registries
    # do not init the default scope here because it will be init in the runner
    register_all_modules(init_default_scope=False)

    # load config
    cfg = mmengine.Config.fromfile(args.config)

    # Remove the original test metrics.
    cfg.test_evaluator = []

    # Load the specified checkpoint.
    cfg.load_from = args.checkpoint

    # build the runner from config
    with tempfile.TemporaryDirectory() as tmpdir:
        cfg.work_dir = tmpdir
        runner = Runner.from_cfg(cfg)
        runner._test_evaluator = [
            SaveConfusionMatrix(args.out, args.show, args.show_dir)
        ]
        runner.test()


if __name__ == '__main__':
    main()
