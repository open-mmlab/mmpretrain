_base_ = [
    '../_base_/models/vit-base-p32.py',
    '../_base_/datasets/imagenet_bs64_clip_224.py',
    '../_base_/schedules/imagenet_bs4096_AdamW.py',
    '../_base_/default_runtime.py'
]

# model setting/mnt/lustre/lirongjie/tmp/clip_ckpt/trans_ckpt
model = dict(backbone=dict(pre_norm=True, ), )

# schedule setting
optim_wrapper = dict(clip_grad=dict(max_norm=1.0))
