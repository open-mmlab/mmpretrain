_base_ = [
    '../_base_/models/gpvit/gpvit_l4.py',
    '../_base_/datasets/imagenet_bs64_swin_224.py',
    '../_base_/schedules/imagenet_bs1024_adamw_swin.py',
    '../_base_/default_runtime.py'
]

# dataset settings
train_dataloader = dict(batch_size=128)

# schedule settings
optim_wrapper = dict(clip_grad=dict(max_norm=1.0))
