_base_ = 'simsiam_resnet50_8xb32-coslr-200e_in1k.py'

# learning rate scheduler
param_scheduler = [
    dict(type='CosineAnnealingLR', T_max=100, by_epoch=True, begin=0, end=100)
]

# runtime settings
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=100)
