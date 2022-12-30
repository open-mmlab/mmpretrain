_base_ = './repvgg-B3_8xb32_in1k.py'

model = dict(backbone=dict(arch='D2se'), head=dict(in_channels=2560))

param_scheduler = [
    # warm up learning rate scheduler
    dict(
        type='LinearLR',
        start_factor=0.0001,
        by_epoch=True,
        begin=0,
        end=5,
        # update by iter
        convert_to_iter_based=True),
    # main learning rate scheduler
    dict(
        type='CosineAnnealingLR',
        T_max=295,
        eta_min=1.0e-6,
        by_epoch=True,
        begin=5,
        end=300)
]

train_cfg = dict(by_epoch=True, max_epochs=300)

default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', interval=1, max_keep_ckpts=3))
