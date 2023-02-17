_base_ = [
    '../_base_/models/barlowtwins.py',
    '../_base_/datasets/imagenet_byol.py',
    '../_base_/schedules/lars_coslr-200e_in1k.py',
    '../_base_/default_runtime.py',
]

train_dataloader = dict(batch_size=256)

# optimizer
optimizer = dict(type='LARS', lr=1.6, momentum=0.9, weight_decay=1e-6)
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=optimizer,
    paramwise_cfg=dict(
        custom_keys={
            'bn': dict(decay_mult=0, lr_mult=0.024, lars_exclude=True),
            'bias': dict(decay_mult=0, lr_mult=0.024, lars_exclude=True),
            # bn layer in ResNet block downsample module
            'downsample.1': dict(
                decay_mult=0, lr_mult=0.024, lars_exclude=True),
        }))

# learning rate scheduler
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=1.6e-4,
        by_epoch=True,
        begin=0,
        end=10,
        convert_to_iter_based=True),
    dict(
        type='CosineAnnealingLR',
        T_max=290,
        eta_min=0.0016,
        by_epoch=True,
        begin=10,
        end=300,
        convert_to_iter_based=True)
]

# runtime settings
default_hooks = dict(checkpoint=dict(max_keep_ckpts=3))

train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=300)
