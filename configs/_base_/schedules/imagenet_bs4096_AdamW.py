# specific to vit pretrain
paramwise_cfg = dict(custom_keys={
    '.cls_token': dict(decay_mult=0.0),
    '.pos_embed': dict(decay_mult=0.0)
})

# optimizer
optimizer = dict(
    type='AdamW',
    lr=0.003,
    weight_decay=0.3,
    paramwise_cfg=paramwise_cfg,
)

# learning policy
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=1e-4,
        by_epoch=False,
        begin=0,
        end=30 * 308),
    dict(
        type='CosineAnnealingLR', T_max=270, by_epoch=True, begin=30, end=300)
]
# old learning policy
# lr_config = dict(
#     policy='CosineAnnealing',
#     min_lr=0,
#     warmup='linear',
#     warmup_iters=10000,
#     warmup_ratio=1e-4,
# )

# train, val, test setting
train_cfg = dict(by_epoch=True, max_epochs=300)
val_cfg = dict(interval=1)  # validate every epoch
test_cfg = dict()
