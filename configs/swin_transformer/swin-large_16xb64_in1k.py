_base_ = [
    '../_base_/models/swin_transformer/large_224.py',
    '../_base_/datasets/imagenet_bs64_swin_224.py',
    '../_base_/schedules/imagenet_bs1024_adamw_swin.py',
    '../_base_/default_runtime.py'
]

# schedule settings
optim_wrapper = dict(clip_grad=dict(max_norm=5.0))

# NOTE: `auto_scale_lr` is for automatically scaling LR,
# USER SHOULD NOT CHANGE ITS VALUES.
# base_batch_size = (16 GPUs) x (64 samples per GPU)
auto_scale_lr = dict(base_batch_size=1024)
