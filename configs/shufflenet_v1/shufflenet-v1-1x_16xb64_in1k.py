_base_ = [
    '../_base_/models/shufflenet_v1_1x.py',
    '../_base_/datasets/imagenet_bs64_pil_resize.py',
    '../_base_/schedules/imagenet_bs1024_linearlr_bn_nowd.py',
    '../_base_/default_runtime.py'
]

# NOTE: `auto_scale_lr` is for automatically scaling LR,
# USER SHOULD NOT CHANGE ITS VALUES.
# base_batch_size = (16 GPUs) x (64 samples per GPU)
auto_scale_lr = dict(base_batch_size=1024)
