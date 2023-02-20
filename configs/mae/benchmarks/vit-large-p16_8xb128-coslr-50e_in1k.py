_base_ = 'vit-base-p16_8xb128-coslr-100e_in1k.py'

# model settings
# MAE ViT-large set drop_path_rate to 0.2
model = dict(
    backbone=dict(arch='large', drop_path_rate=0.1),
    head=dict(in_channels=1024))

# optimizer settings
# learning rate and layer decay rate are set to 0.004 and 0.75 respectively
optim_wrapper = dict(optimizer=dict(lr=0.004, layer_decay_rate=0.75))

# learning rate scheduler
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=1e-4,
        by_epoch=True,
        begin=0,
        end=5,
        convert_to_iter_based=True),
    dict(
        type='CosineAnnealingLR',
        T_max=45,
        by_epoch=True,
        begin=5,
        end=50,
        eta_min=1e-6,
        convert_to_iter_based=True)
]

# training cfg
# fine-tuning for 50 epochs for ViT-large
train_cfg = dict(max_epochs=50)
