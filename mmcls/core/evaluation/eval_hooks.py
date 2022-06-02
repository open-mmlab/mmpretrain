# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp

import torch.distributed as dist
from mmcv.runner import DistEvalHook as BaseDistEvalHook
from mmcv.runner import EvalHook as BaseEvalHook
from torch.nn.modules.batchnorm import _BatchNorm


class EvalHook(BaseEvalHook):
    """Non-Distributed evaluation hook.

    Comparing with the ``EvalHook`` in MMCV, this hook will save the latest
    evaluation results as an attribute for other hooks to use (like
    `MMClsWandbHook`).
    """

    def __init__(self, dataloader, **kwargs):
        super(EvalHook, self).__init__(dataloader, **kwargs)
        self.latest_results = None

    def _do_evaluate(self, runner):
        """perform evaluation and save ckpt."""
        results = self.test_fn(runner.model, self.dataloader)
        self.latest_results = results
        runner.log_buffer.output['eval_iter_num'] = len(self.dataloader)
        key_score = self.evaluate(runner, results)
        # the key_score may be `None` so it needs to skip the action to save
        # the best checkpoint
        if self.save_best and key_score:
            self._save_ckpt(runner, key_score)


class DistEvalHook(BaseDistEvalHook):
    """Non-Distributed evaluation hook.

    Comparing with the ``EvalHook`` in MMCV, this hook will save the latest
    evaluation results as an attribute for other hooks to use (like
    `MMClsWandbHook`).
    """

    def __init__(self, dataloader, **kwargs):
        super(DistEvalHook, self).__init__(dataloader, **kwargs)
        self.latest_results = None

    def _do_evaluate(self, runner):
        """perform evaluation and save ckpt."""
        # Synchronization of BatchNorm's buffer (running_mean
        # and running_var) is not supported in the DDP of pytorch,
        # which may cause the inconsistent performance of models in
        # different ranks, so we broadcast BatchNorm's buffers
        # of rank 0 to other ranks to avoid this.
        if self.broadcast_bn_buffer:
            model = runner.model
            for name, module in model.named_modules():
                if isinstance(module,
                              _BatchNorm) and module.track_running_stats:
                    dist.broadcast(module.running_var, 0)
                    dist.broadcast(module.running_mean, 0)

        tmpdir = self.tmpdir
        if tmpdir is None:
            tmpdir = osp.join(runner.work_dir, '.eval_hook')

        results = self.test_fn(
            runner.model,
            self.dataloader,
            tmpdir=tmpdir,
            gpu_collect=self.gpu_collect)
        self.latest_results = results
        if runner.rank == 0:
            print('\n')
            runner.log_buffer.output['eval_iter_num'] = len(self.dataloader)
            key_score = self.evaluate(runner, results)
            # the key_score may be `None` so it needs to skip the action to
            # save the best checkpoint
            if self.save_best and key_score:
                self._save_ckpt(runner, key_score)
