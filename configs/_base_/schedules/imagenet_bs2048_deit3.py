# optimizer
optim_wrapper = dict(
    optimizer=dict(
        type='Lamb', lr=0.003, weight_decay=0.05, max_grad_norm=1.0, eps=1e-8),
    # specific to vit pretrain
    paramwise_cfg=dict(
        norm_decay_mult=0.0,
        bias_decay_mult=0.0,
        flat_decay_mult=0.0,
        custom_keys={
            '.cls_token': dict(decay_mult=0.0),
            '.pos_embed': dict(decay_mult=0.0),
        }),
)

# learning policy
param_scheduler = [
    # warm up learning rate scheduler
    dict(
        type='LinearLR',
        start_factor=1e-6 / 0.003,
        by_epoch=True,
        begin=0,
        end=5,
        # update by iter
        convert_to_iter_based=False),
    # main learning rate scheduler
    dict(type='CosineAnnealingLR', by_epoch=True, begin=5)
]

# train, val, test setting
train_cfg = dict(by_epoch=True, max_epochs=800, val_interval=1)
val_cfg = dict()
test_cfg = dict()

# NOTE: `auto_scale_lr` is for automatically scaling LR,
# based on the actual training batch size.
auto_scale_lr = dict(base_batch_size=2048)
