_base_ = [
    '../_base_/models/vit-large-p16.py',
    '../_base_/datasets/imagenet_bs64_pil_resize_autoaug.py',
    '../_base_/schedules/imagenet_bs4096_AdamW.py',
    '../_base_/default_runtime.py'
]

default_hooks = dict(optimizer=dict(grad_clip=dict(max_norm=1.0)))

model = dict(
    head=dict(hidden_dim=3072),
    train_cfg=dict(
        augments=dict(type='BatchMixup', alpha=0.2, num_classes=1000,
                      prob=1.)))
