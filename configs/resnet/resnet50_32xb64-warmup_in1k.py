_base_ = [
    '../_base_/models/resnet50.py', '../_base_/datasets/imagenet_bs64.py',
    '../_base_/schedules/imagenet_bs2048.py', '../_base_/default_runtime.py'
]

# NOTE: `auto_scale_lr` is for automatically scaling LR,
# USER SHOULD NOT CHANGE ITS VALUES.
# base_batch_size = (32 GPUs) x (64 samples per GPU)
auto_scale_lr = dict(base_batch_size=2048)
