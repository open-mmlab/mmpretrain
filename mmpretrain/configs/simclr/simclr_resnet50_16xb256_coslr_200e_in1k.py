if '_base_':
    from .._base_.datasets.imagenet_bs32_simclr import *
    from .._base_.schedules.imagenet_lars_coslr_200e import *
    from .._base_.default_runtime import *
from mmpretrain.models.selfsup.simclr import SimCLR
from mmpretrain.models.backbones.resnet import ResNet
from mmpretrain.models.necks.nonlinear_neck import NonLinearNeck
from mmpretrain.models.heads.contrastive_head import ContrastiveHead
from mmpretrain.models.losses.cross_entropy_loss import CrossEntropyLoss
from mmengine.optim.optimizer.optimizer_wrapper import OptimWrapper
from mmpretrain.engine.optimizers.lars import LARS
from mmengine.hooks.checkpoint_hook import CheckpointHook

# dataset settings
train_dataloader.merge(dict(batch_size=256))

# model settings
model = dict(
    type=SimCLR,
    backbone=dict(
        type=ResNet,
        depth=50,
        norm_cfg=dict(type='SyncBN'),
        zero_init_residual=True),
    neck=dict(
        type=NonLinearNeck,  # SimCLR non-linear neck
        in_channels=2048,
        hid_channels=2048,
        out_channels=128,
        num_layers=2,
        with_avg_pool=True),
    head=dict(
        type=ContrastiveHead,
        loss=dict(type=CrossEntropyLoss),
        temperature=0.1),
)

# optimizer
optim_wrapper.merge(
    dict(
        type=OptimWrapper,
        optimizer=dict(type=LARS, lr=4.8, momentum=0.9, weight_decay=1e-6),
        paramwise_cfg=dict(
            custom_keys={
                'bn': dict(decay_mult=0, lars_exclude=True),
                'bias': dict(decay_mult=0, lars_exclude=True),
                # bn layer in ResNet block downsample module
                'downsample.1': dict(decay_mult=0, lars_exclude=True),
            })))

# runtime settings
default_hooks.merge(
    dict(
        # only keeps the latest 3 checkpoints
        checkpoint=dict(type=CheckpointHook, interval=10, max_keep_ckpts=3)))
