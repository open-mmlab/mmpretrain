_base_ = [
    '../_base_/models/mlp_mixer_large_patch16.py',
    '../_base_/datasets/imagenet_bs64_mixer_224.py',
    '../_base_/schedules/imagenet_bs4096_AdamW.py',
    '../_base_/default_runtime.py',
]

optim_wrapper = dict(clip_grad=dict(max_norm=1.0))

# NOTE: `auto_scale_lr` is for automatically scaling LR,
# USER SHOULD NOT CHANGE ITS VALUES.
# base_batch_size = (64 GPUs) x (64 samples per GPU)
auto_scale_lr = dict(base_batch_size=4096)
