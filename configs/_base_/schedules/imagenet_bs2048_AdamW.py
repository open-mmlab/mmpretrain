# optimizer
# In ClassyVision, the lr is set to 0.003 for bs4096.
# In this implementation(bs2048), lr = 0.003 / 4096 * (32bs * 64gpus) = 0.0015
optimizer = dict(type='AdamW', lr=0.0015, weight_decay=0.3)

# specific to vit pretrain
paramwise_cfg = dict(
    custom_keys={
        '.backbone.cls_token': dict(decay_mult=0.0),
        '.backbone.pos_embed': dict(decay_mult=0.0)
    })
# learning policy
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=1e-3,
        by_epoch=False,
        begin=0,
        end=10 * 626),
    dict(
        type='CosineAnnealingLR',
        T_max=290,
        eta_min=1e-2,
        by_epoch=True,
        begin=10,
        end=300)
]
# old learning policy
# lr_config = dict(
#     policy='CosineAnnealing',
#     min_lr=0,
#     warmup='linear',
#     warmup_iters=10000,
#     warmup_ratio=1e-4)

# train, val, test setting
train_cfg = dict(by_epoch=True, max_epochs=300)
val_cfg = dict(interval=1)  # validate every epoch
test_cfg = dict()
