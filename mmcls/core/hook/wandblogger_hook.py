# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp

import numpy as np
from mmcv.runner import HOOKS, BaseRunner
from mmcv.runner.dist_utils import get_dist_info, master_only
from mmcv.runner.hooks.checkpoint import CheckpointHook
from mmcv.runner.hooks.evaluation import DistEvalHook, EvalHook
from mmcv.runner.hooks.logger.wandb import WandbLoggerHook


@HOOKS.register_module()
class MMClsWandbHook(WandbLoggerHook):
    """Enhanced Wandb logger hook for classification.

    Comparing with the :cls:`mmcv.runner.WandbLoggerHook`, this hook can not
    only automatically log all information in ``log_buffer`` but also log
    the following extra information.

    - **Checkpoints**: If ``log_checkpoint`` is True, the checkpoint saved at
      every checkpoint interval will be saved as W&B Artifacts. This depends on
      the : class:`mmcv.runner.CheckpointHook` whose priority is higher than
      this hook. Please refer to
      https://docs.wandb.ai/guides/artifacts/model-versioning to learn more
      about model versioning with W&B Artifacts.

    - **Checkpoint Metadata**: If ``log_checkpoint_metadata`` is True, every
      checkpoint artifact will have a metadata associated with it. The metadata
      contains the evaluation metrics computed on validation data with that
      checkpoint along with the current epoch/iter. It depends on
      :class:`EvalHook` whose priority is higher than this hook.

    - **Evaluation**: At every interval, this hook logs the model prediction as
      interactive W&B Tables. The number of samples logged is given by
      ``num_eval_images``. Currently, this hook logs the predicted labels along
      with the ground truth at every evaluation interval. This depends on the
      :class:`EvalHook` whose priority is higher than this hook. Also note that
      the data is just logged once and subsequent evaluation tables uses
      reference to the logged data to save memory usage. Please refer to
      https://docs.wandb.ai/guides/data-vis to learn more about W&B Tables.

    Here is a config example:

    .. code:: python

        checkpoint_config = dict(interval=10)

        # To log checkpoint metadata, the interval of checkpoint saving should
        # be divisible by the interval of evaluation.
        evaluation = dict(interval=5)

        log_config = dict(
            ...
            hooks=[
                ...
                dict(type='MMClsWandbHook',
                     init_kwargs={
                         'entity': "YOUR_ENTITY",
                         'project': "YOUR_PROJECT_NAME"
                     },
                     log_checkpoint=True,
                     log_checkpoint_metadata=True,
                     num_eval_images=100)
            ])

    Args:
        init_kwargs (dict): A dict passed to wandb.init to initialize
            a W&B run. Please refer to https://docs.wandb.ai/ref/python/init
            for possible key-value pairs.
        interval (int): Logging interval (every k iterations). Defaults to 10.
        log_checkpoint (bool): Save the checkpoint at every checkpoint interval
            as W&B Artifacts. Use this for model versioning where each version
            is a checkpoint. Defaults to False.
        log_checkpoint_metadata (bool): Log the evaluation metrics computed
            on the validation data with the checkpoint, along with current
            epoch as a metadata to that checkpoint.
            Defaults to True.
        num_eval_images (int): The number of validation images to be logged.
            If zero, the evaluation won't be logged. Defaults to 100.
    """

    def __init__(self,
                 init_kwargs=None,
                 interval=10,
                 log_checkpoint=False,
                 log_checkpoint_metadata=False,
                 num_eval_images=100,
                 **kwargs):
        super(MMClsWandbHook, self).__init__(init_kwargs, interval, **kwargs)

        self.log_checkpoint = log_checkpoint
        self.log_checkpoint_metadata = (
            log_checkpoint and log_checkpoint_metadata)
        self.num_eval_images = num_eval_images
        self.log_evaluation = (num_eval_images > 0)
        self.ckpt_hook: CheckpointHook = None
        self.eval_hook: EvalHook = None

    @master_only
    def before_run(self, runner: BaseRunner):
        super(MMClsWandbHook, self).before_run(runner)

        # Inspect CheckpointHook and EvalHook
        for hook in runner.hooks:
            if isinstance(hook, CheckpointHook):
                self.ckpt_hook = hook
            if isinstance(hook, (EvalHook, DistEvalHook)):
                self.eval_hook = hook

        # Check conditions to log checkpoint
        if self.log_checkpoint:
            if self.ckpt_hook is None:
                self.log_checkpoint = False
                self.log_checkpoint_metadata = False
                runner.logger.warning(
                    'To log checkpoint in MMClsWandbHook, `CheckpointHook` is'
                    'required, please check hooks in the runner.')
            else:
                self.ckpt_interval = self.ckpt_hook.interval

        # Check conditions to log evaluation
        if self.log_evaluation or self.log_checkpoint_metadata:
            if self.eval_hook is None:
                self.log_evaluation = False
                self.log_checkpoint_metadata = False
                runner.logger.warning(
                    'To log evaluation or checkpoint metadata in '
                    'MMClsWandbHook, `EvalHook` or `DistEvalHook` in mmcls '
                    'is required, please check whether the validation '
                    'is enabled.')
            else:
                self.eval_interval = self.eval_hook.interval
            self.val_dataset = self.eval_hook.dataloader.dataset
            if (self.log_evaluation
                    and self.num_eval_images > len(self.val_dataset)):
                self.num_eval_images = len(self.val_dataset)
                runner.logger.warning(
                    f'The num_eval_images ({self.num_eval_images}) is '
                    'greater than the total number of validation samples '
                    f'({len(self.val_dataset)}). The complete validation '
                    'dataset will be logged.')

        # Check conditions to log checkpoint metadata
        if self.log_checkpoint_metadata:
            assert self.ckpt_interval % self.eval_interval == 0, \
                'To log checkpoint metadata in MMClsWandbHook, the interval ' \
                f'of checkpoint saving ({self.ckpt_interval}) should be ' \
                'divisible by the interval of evaluation ' \
                f'({self.eval_interval}).'

        # Initialize evaluation table
        if self.log_evaluation:
            # Initialize data table
            self._init_data_table()
            # Add ground truth to the data table
            self._add_ground_truth()
            # Log ground truth data
            self._log_data_table()

    @master_only
    def after_train_epoch(self, runner):
        super(MMClsWandbHook, self).after_train_epoch(runner)

        if not self.by_epoch:
            return

        # Save checkpoint and metadata
        if (self.log_checkpoint
                and self.every_n_epochs(runner, self.ckpt_interval)
                or (self.ckpt_hook.save_last and self.is_last_epoch(runner))):
            if self.log_checkpoint_metadata and self.eval_hook:
                metadata = {
                    'epoch': runner.epoch + 1,
                    **self._get_eval_results()
                }
            else:
                metadata = None
            aliases = [f'epoch_{runner.epoch+1}', 'latest']
            model_path = osp.join(self.ckpt_hook.out_dir,
                                  f'epoch_{runner.epoch+1}.pth')
            self._log_ckpt_as_artifact(model_path, aliases, metadata)

        # Save prediction table
        if self.log_evaluation and self.eval_hook._should_evaluate(runner):
            results = self.eval_hook.latest_results
            # Initialize evaluation table
            self._init_pred_table()
            # Add predictions to evaluation table
            self._add_predictions(results, runner.epoch + 1)
            # Log the evaluation table
            self._log_eval_table(runner.epoch + 1)

    def after_train_iter(self, runner):
        if self.get_mode(runner) == 'train':
            # An ugly patch. The iter-based eval hook will call the
            # `after_train_iter` method of all logger hooks before evaluation.
            # Use this trick to skip that call.
            # Don't call super method at first, it will clear the log_buffer
            return super(MMClsWandbHook, self).after_train_iter(runner)
        else:
            super(MMClsWandbHook, self).after_train_iter(runner)

        rank, _ = get_dist_info()
        if rank != 0:
            return

        if self.by_epoch:
            return

        # Save checkpoint and metadata
        if (self.log_checkpoint
                and self.every_n_iters(runner, self.ckpt_interval)
                or (self.ckpt_hook.save_last and self.is_last_iter(runner))):
            if self.log_checkpoint_metadata and self.eval_hook:
                metadata = {
                    'iter': runner.iter + 1,
                    **self._get_eval_results()
                }
            else:
                metadata = None
            aliases = [f'iter_{runner.iter+1}', 'latest']
            model_path = osp.join(self.ckpt_hook.out_dir,
                                  f'iter_{runner.iter+1}.pth')
            self._log_ckpt_as_artifact(model_path, aliases, metadata)

        # Save prediction table
        if self.log_evaluation and self.eval_hook._should_evaluate(runner):
            results = self.eval_hook.latest_results
            # Initialize evaluation table
            self._init_pred_table()
            # Log predictions
            self._add_predictions(results, runner.iter + 1)
            # Log the table
            self._log_eval_table(runner.iter + 1)

    @master_only
    def after_run(self, runner):
        self.wandb.finish()

    def _log_ckpt_as_artifact(self, model_path, aliases, metadata=None):
        """Log model checkpoint as  W&B Artifact.

        Args:
            model_path (str): Path of the checkpoint to log.
            aliases (list): List of the aliases associated with this artifact.
            metadata (dict, optional): Metadata associated with this artifact.
        """
        model_artifact = self.wandb.Artifact(
            f'run_{self.wandb.run.id}_model', type='model', metadata=metadata)
        model_artifact.add_file(model_path)
        self.wandb.log_artifact(model_artifact, aliases=aliases)

    def _get_eval_results(self):
        """Get model evaluation results."""
        results = self.eval_hook.latest_results
        eval_results = self.val_dataset.evaluate(
            results, logger='silent', **self.eval_hook.eval_kwargs)
        return eval_results

    def _init_data_table(self):
        """Initialize the W&B Tables for validation data."""
        columns = ['image_name', 'image', 'ground_truth']
        self.data_table = self.wandb.Table(columns=columns)

    def _init_pred_table(self):
        """Initialize the W&B Tables for model evaluation."""
        columns = ['epoch'] if self.by_epoch else ['iter']
        columns += ['image_name', 'image', 'ground_truth', 'prediction'
                    ] + list(self.val_dataset.CLASSES)
        self.eval_table = self.wandb.Table(columns=columns)

    def _add_ground_truth(self):
        # Get image loading pipeline
        from mmcls.datasets.pipelines import LoadImageFromFile
        img_loader = None
        for t in self.val_dataset.pipeline.transforms:
            if isinstance(t, LoadImageFromFile):
                img_loader = t

        CLASSES = self.val_dataset.CLASSES
        self.eval_image_indexs = np.arange(len(self.val_dataset))
        # Set seed so that same validation set is logged each time.
        np.random.seed(42)
        np.random.shuffle(self.eval_image_indexs)
        self.eval_image_indexs = self.eval_image_indexs[:self.num_eval_images]

        for idx in self.eval_image_indexs:
            img_info = self.val_dataset.data_infos[idx]
            if img_loader is not None:
                img_info = img_loader(img_info)
                # Get image and convert from BGR to RGB
                image = img_info['img'][..., ::-1]
            else:
                # For CIFAR dataset.
                image = img_info['img']
            image_name = img_info.get('filename', f'img_{idx}')
            gt_label = img_info.get('gt_label').item()

            self.data_table.add_data(image_name, self.wandb.Image(image),
                                     CLASSES[gt_label])

    def _add_predictions(self, results, idx):
        table_idxs = self.data_table_ref.get_index()
        assert len(table_idxs) == len(self.eval_image_indexs)

        for ndx, eval_image_index in enumerate(self.eval_image_indexs):
            result = results[eval_image_index]

            self.eval_table.add_data(
                idx, self.data_table_ref.data[ndx][0],
                self.data_table_ref.data[ndx][1],
                self.data_table_ref.data[ndx][2],
                self.val_dataset.CLASSES[np.argmax(result)], *tuple(result))

    def _log_data_table(self):
        """Log the W&B Tables for validation data as artifact and calls
        `use_artifact` on it so that the evaluation table can use the reference
        of already uploaded images.

        This allows the data to be uploaded just once.
        """
        data_artifact = self.wandb.Artifact('val', type='dataset')
        data_artifact.add(self.data_table, 'val_data')

        self.wandb.run.use_artifact(data_artifact)
        data_artifact.wait()

        self.data_table_ref = data_artifact.get('val_data')

    def _log_eval_table(self, idx):
        """Log the W&B Tables for model evaluation.

        The table will be logged multiple times creating new version. Use this
        to compare models at different intervals interactively.
        """
        pred_artifact = self.wandb.Artifact(
            f'run_{self.wandb.run.id}_pred', type='evaluation')
        pred_artifact.add(self.eval_table, 'eval_data')
        if self.by_epoch:
            aliases = ['latest', f'epoch_{idx}']
        else:
            aliases = ['latest', f'iter_{idx}']
        self.wandb.run.log_artifact(pred_artifact, aliases=aliases)
