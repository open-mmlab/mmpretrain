_base_ = [
    '../_base_/models/mobilenet-v3-small_8xb16_cifar.py',
    '../_base_/datasets/cifar10_bs16.py',
    '../_base_/schedules/cifar10_bs128.py', '../_base_/default_runtime.py'
]

# schedule settings
param_scheduler = dict(
    type='MultiStepLR',
    by_epoch=True,
    milestones=[120, 170],
    gamma=0.1,
)

train_cfg = dict(by_epoch=True, max_epochs=200)

# NOTE: `auto_scale_lr` is for automatically scaling LR,
# USER SHOULD NOT CHANGE ITS VALUES.
# base_batch_size = (8 GPUs) x (16 samples per GPU)
auto_scale_lr = dict(base_batch_size=128)
