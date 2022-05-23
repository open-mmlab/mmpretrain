# optimizer
optimizer = dict(type='SGD', lr=0.045, momentum=0.9, weight_decay=0.00004)
# learning policy
lr_config = dict(policy='step', gamma=0.98, step=1)
runner = dict(type='EpochBasedRunner', max_epochs=300)
