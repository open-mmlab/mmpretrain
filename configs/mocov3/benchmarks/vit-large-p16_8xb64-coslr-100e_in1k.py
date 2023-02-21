_base_ = 'vit-base-p16_8xb64-coslr-150e_in1k.py'

# model settings
model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='VisionTransformer',
        arch='large',
        img_size=224,
        patch_size=16,
        drop_path_rate=0.5,
    ),
    neck=None,
    head=dict(
        type='VisionTransformerClsHead',
        num_classes=1000,
        in_channels=1024,
        loss=dict(
            type='LabelSmoothLoss', label_smooth_val=0.1, mode='original'),
    ))

# learning rate scheduler
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=1e-3,
        begin=0,
        end=5,
        convert_to_iter_based=True),
    dict(
        type='CosineAnnealingLR',
        T_max=95,
        eta_min=1e-5,
        by_epoch=True,
        begin=5,
        end=100,
        convert_to_iter_based=True)
]

# runtime settings
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=100)
