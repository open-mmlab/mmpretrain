_base_ = [
    '../_base_/models/t2t-vit-t-14.py',
    '../_base_/datasets/imagenet_bs64_t2t_224.py',
    '../_base_/default_runtime.py',
]

# schedule settings
optim_wrapper = dict(
    optimizer=dict(type='AdamW', lr=5e-4, weight_decay=0.05),
    paramwise_cfg=dict(
        norm_decay_mult=0.0,
        bias_decay_mult=0.0,
        custom_keys={'cls_token': dict(decay_mult=0.0)},
    ),
)

param_scheduler = [
    # warm up learning rate scheduler
    dict(
        type='LinearLR',
        start_factor=1e-6,
        by_epoch=True,
        begin=0,
        end=10,
        # update by iter
        convert_to_iter_based=True),
    # main learning rate scheduler
    dict(
        type='CosineAnnealingLR',
        T_max=290,
        eta_min=1e-5,
        by_epoch=True,
        begin=10,
        end=300),
    # cool down learning rate scheduler
    dict(type='ConstantLR', factor=0.1, by_epoch=True, begin=300, end=310),
]

train_cfg = dict(by_epoch=True, max_epochs=310, val_interval=1)
val_cfg = dict()
test_cfg = dict()

# runtime settings
custom_hooks = [dict(type='EMAHook', momentum=4e-5, priority='ABOVE_NORMAL')]

# NOTE: `auto_scale_lr` is for automatically scaling LR
# based on the actual training batch size.
# base_batch_size = (8 GPUs) x (64 samples per GPU)
auto_scale_lr = dict(base_batch_size=512)
