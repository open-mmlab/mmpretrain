_base_ = [
    '../_base_/models/mobilenet_v3/mobilenet_v3_small_cifar.py',
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
