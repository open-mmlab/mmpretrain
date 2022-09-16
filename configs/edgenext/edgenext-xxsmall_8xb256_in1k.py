_base_ = [
    '../_base_/datasets/imagenet_bs64_edgenext_256.py',
    '../_base_/schedules/imagenet_bs1024_adamw_swin.py',
    '../_base_/default_runtime.py',
]

# schedule setting
optim_wrapper = dict(
    optimizer=dict(lr=6e-3),
    clip_grad=dict(max_norm=5.0),
)

# runtime setting
custom_hooks = [dict(type='EMAHook', momentum=4e-5, priority='ABOVE_NORMAL')]

# NOTE: `auto_scale_lr` is for automatically scaling LR
# based on the actual training batch size.
# base_batch_size = (32 GPUs) x (128 samples per GPU)
auto_scale_lr = dict(base_batch_size=4096)

# Model settings
model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='EdgeNeXt',
        arch='xxsmall',
        out_indices=(3, ),
        drop_path_rate=0.1,
        gap_before_final_norm=True,
        init_cfg=[
            dict(
                type='TruncNormal',
                layer=['Conv2d', 'Linear'],
                std=.02,
                bias=0.),
            dict(type='Constant', layer=['LayerNorm'], val=1., bias=0.),
        ]),
    head=dict(
        type='LinearClsHead',
        num_classes=1000,
        in_channels=168,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
    ))
