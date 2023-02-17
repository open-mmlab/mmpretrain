# optimizer wrapper
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr=0.03, weight_decay=1e-4, momentum=0.9))

# learning rate scheduler
param_scheduler = [
    dict(type='MultiStepLR', by_epoch=True, milestones=[120, 160], gamma=0.1)
]

# runtime settings
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=200)
