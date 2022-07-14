_base_ = [
    '../_base_/models/vit-base-p16.py',
    '../_base_/datasets/imagenet_bs64_pil_resize_autoaug.py',
    '../_base_/schedules/imagenet_bs4096_AdamW.py',
    '../_base_/default_runtime.py'
]

# model setting
model = dict(
    head=dict(hidden_dim=3072),
    train_cfg=dict(augments=dict(type='Mixup', alpha=0.2, num_classes=1000)),
)

# schedule setting
optim_wrapper = dict(clip_grad=dict(max_norm=1.0))

# NOTE: `auto_scale_lr` is for automatically scaling LR,
# USER SHOULD NOT CHANGE ITS VALUES.
# base_batch_size = (64 GPUs) x (64 samples per GPU)
auto_scale_lr = dict(base_batch_size=4096)
