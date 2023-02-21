_base_ = 'simmim_swin-base-w6_16xb128-amp-coslr-100e_in1k-192px.py'

# optimizer wrapper
optim_wrapper = dict(
    optimizer=dict(
        type='AdamW', lr=1e-4 * 2048 / 512, betas=(0.9, 0.999), eps=1e-8))

# learning rate scheduler
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=5e-7 / 1e-4,
        by_epoch=True,
        begin=0,
        end=10,
        convert_to_iter_based=True),
    dict(
        type='MultiStepLR',
        milestones=[700],
        by_epoch=True,
        begin=10,
        end=800,
        convert_to_iter_based=True)
]

# schedule
train_cfg = dict(max_epochs=800)
