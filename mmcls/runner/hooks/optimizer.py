# Copyright (c) Open-MMLab. All rights reserved.
from distutils.version import LooseVersion

from mmcv.runner.dist_utils import allreduce_grads
from mmcv.runner.hooks import Fp16OptimizerHook as _Fp16OptimizerHook
from mmcv.runner.hooks import OptimizerHook as _OptimizerHook
from mmcv.utils import TORCH_VERSION

from .builder import HOOKS


@HOOKS.register_module()
class OptimizerHook(_OptimizerHook):

    def __init__(self, grad_clip=None, accumulation_step=1, *args, **kwargs):
        super(OptimizerHook, self).__init__(grad_clip, *args, **kwargs)
        self.accumulation_step = accumulation_step

    def after_train_iter(self, runner):
        loss = runner.outputs['loss']
        loss = loss / self.accumulation_step
        loss.backward()

        # If in accumulation, only do backward.
        if not self.every_n_iters(runner, self.accumulation_step):
            return

        # grad norm
        if self.grad_clip is not None:
            grad_norm = self.clip_grads(runner.model.parameters())
            if grad_norm is not None:
                # Add grad norm to the logger
                runner.log_buffer.update({'grad_norm': float(grad_norm)},
                                         runner.outputs['num_samples'])

        runner.optimizer.step()
        runner.optimizer.zero_grad()


if (TORCH_VERSION != 'parrots'
        and LooseVersion(TORCH_VERSION) >= LooseVersion('1.6.0')):

    @HOOKS.register_module()
    class Fp16OptimizerHook(_Fp16OptimizerHook):
        """FP16 optimizer hook (using PyTorch's implementation).

        If you are using PyTorch >= 1.6, torch.cuda.amp is used as the backend,
        to take care of the optimization procedure.

        Args:
            loss_scale (float | str | dict): Scale factor configuration.
                If loss_scale is a float, static loss scaling will be used with
                the specified scale. If loss_scale is a string, it must be
                'dynamic', then dynamic loss scaling will be used.
                It can also be a dict containing arguments of GradScalar.
                Defaults to 512. For Pytorch >= 1.6, mmcv uses official
                implementation of GradScaler. If you use a dict version of
                loss_scale to create GradScaler, please refer to:
                https://pytorch.org/docs/stable/amp.html#torch.cuda.amp.GradScaler
                for the parameters.

        Examples:
            >>> loss_scale = dict(
            ...     init_scale=65536.0,
            ...     growth_factor=2.0,
            ...     backoff_factor=0.5,
            ...     growth_interval=2000
            ... )
            >>> optimizer_hook = Fp16OptimizerHook(loss_scale=loss_scale)
        """

        def __init__(self,
                     grad_clip=None,
                     accumulation_step=1,
                     coalesce=True,
                     bucket_size_mb=-1,
                     loss_scale=512.,
                     distributed=True,
                     *args,
                     **kwargs):
            super(Fp16OptimizerHook, self).__init__(
                grad_clip=grad_clip,
                coalesce=coalesce,
                bucket_size_mb=bucket_size_mb,
                loss_scale=loss_scale,
                distributed=distributed,
                *args,
                **kwargs)
            self.accumulation_step = accumulation_step

        def after_train_iter(self, runner):
            """Backward optimization steps for Mixed Precision Training. For
            dynamic loss scaling, please refer to
            https://pytorch.org/docs/stable/amp.html#torch.cuda.amp.GradScaler.

            1. Scale the loss by a scale factor.
            2. Backward the loss to obtain the gradients.
            3. Unscale the optimizer’s gradient tensors.
            4. Call optimizer.step() and update scale factor.
            5. Save loss_scaler state_dict for resume purpose.
            """
            loss = runner.outputs['loss']
            loss = loss / self.accumulation_step

            # scale the loss value
            self.loss_scaler.scale(loss).backward()

            # If in accumulation, only do backward.
            if not self.every_n_iters(runner, self.accumulation_step):
                return

            # copy fp16 grads in the model to fp32 params in the optimizer
            self.loss_scaler.unscale_(runner.optimizer)

            # grad clip
            if self.grad_clip is not None:
                grad_norm = self.clip_grads(runner.model.parameters())
                if grad_norm is not None:
                    # Add grad norm to the logger
                    runner.log_buffer.update({'grad_norm': float(grad_norm)},
                                             runner.outputs['num_samples'])
            # backward and update scaler
            self.loss_scaler.step(runner.optimizer)
            self.loss_scaler.update(self._scale_update_param)

            # save state_dict of loss_scaler
            runner.meta.setdefault(
                'fp16', {})['loss_scaler'] = self.loss_scaler.state_dict()

            # clear grads
            runner.model.zero_grad()
            runner.optimizer.zero_grad()

else:

    @HOOKS.register_module()
    class Fp16OptimizerHook(_Fp16OptimizerHook):
        """FP16 optimizer hook (mmcv's implementation).

        The steps of fp16 optimizer is as follows.
        1. Scale the loss value.
        2. BP in the fp16 model.
        2. Copy gradients from fp16 model to fp32 weights.
        3. Update fp32 weights.
        4. Copy updated parameters from fp32 weights to fp16 model.

        Refer to https://arxiv.org/abs/1710.03740 for more details.

        Args:
            loss_scale (float | str | dict): Scale factor configuration.
                If loss_scale is a float, static loss scaling will be used with
                the specified scale. If loss_scale is a string, it must be
                'dynamic', then dynamic loss scaling will be used.
                It can also be a dict containing arguments of LossScaler.
                Defaults to 512.
        """

        def __init__(self,
                     grad_clip=None,
                     accumulation_step=1,
                     coalesce=True,
                     bucket_size_mb=-1,
                     loss_scale=512.,
                     distributed=True,
                     *args,
                     **kwargs):
            super(Fp16OptimizerHook, self).__init__(
                grad_clip=grad_clip,
                coalesce=coalesce,
                bucket_size_mb=bucket_size_mb,
                loss_scale=loss_scale,
                distributed=distributed,
                *args,
                **kwargs)
            self.accumulation_step = accumulation_step

        def after_train_iter(self, runner):
            """Backward optimization steps for Mixed Precision Training. For
            dynamic loss scaling, please refer `loss_scalar.py`

            1. Scale the loss by a scale factor.
            2. Backward the loss to obtain the gradients (fp16).
            3. Copy gradients from the model to the fp32 weight copy.
            4. Scale the gradients back and update the fp32 weight copy.
            5. Copy back the params from fp32 weight copy to the fp16 model.
            6. Save loss_scaler state_dict for resume purpose.
            """
            loss = runner.outputs['loss']
            loss = loss / self.accumulation_step

            # scale the loss value
            scaled_loss = runner.outputs['loss'] * self.loss_scaler.loss_scale
            scaled_loss.backward()

            # If in accumulation, only do backward.
            if not self.every_n_iters(runner, self.accumulation_step):
                return

            # copy fp16 grads in the model to fp32 params in the optimizer
            fp32_weights = []
            for param_group in runner.optimizer.param_groups:
                fp32_weights += param_group['params']
            self.copy_grads_to_fp32(runner.model, fp32_weights)
            # allreduce grads
            if self.distributed:
                allreduce_grads(fp32_weights, self.coalesce,
                                self.bucket_size_mb)

            has_overflow = self.loss_scaler.has_overflow(fp32_weights)
            # if has overflow, skip this iteration
            if not has_overflow:
                # scale the gradients back
                for param in fp32_weights:
                    if param.grad is not None:
                        param.grad.div_(self.loss_scaler.loss_scale)
                if self.grad_clip is not None:
                    grad_norm = self.clip_grads(fp32_weights)
                    if grad_norm is not None:
                        # Add grad norm to the logger
                        runner.log_buffer.update(
                            {'grad_norm': float(grad_norm)},
                            runner.outputs['num_samples'])
                # update fp32 params
                runner.optimizer.step()
                # copy fp32 params to the fp16 model
                self.copy_params_to_fp16(runner.model, fp32_weights)
            else:
                runner.logger.warning('Check overflow, downscale loss scale '
                                      f'to {self.loss_scaler.cur_scale}')

            self.loss_scaler.update_scale(has_overflow)

            # save state_dict of loss_scaler
            runner.meta.setdefault(
                'fp16', {})['loss_scaler'] = self.loss_scaler.state_dict()

            # clear grads
            runner.model.zero_grad()
            runner.optimizer.zero_grad()
