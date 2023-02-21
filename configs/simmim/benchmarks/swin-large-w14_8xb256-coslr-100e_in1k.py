_base_ = 'swin-base-w7_8xb256-coslr-100e_in1k.py'

# model settings
model = dict(
    backbone=dict(
        arch='large',
        img_size=224,
        drop_path_rate=0.2,
        stage_cfgs=dict(block_cfgs=dict(window_size=14)),
        pad_small_map=True),
    head=dict(in_channels=1536))

# schedule settings
optim_wrapper = dict(optimizer=dict(layer_decay_rate=0.7))

# learning rate scheduler
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=2.5e-7 / 1.25e-3,
        by_epoch=True,
        begin=0,
        end=20,
        convert_to_iter_based=True),
    dict(
        type='CosineAnnealingLR',
        T_max=100,
        eta_min=1e-6,
        by_epoch=True,
        begin=20,
        end=100,
        convert_to_iter_based=True)
]
