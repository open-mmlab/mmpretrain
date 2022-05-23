# optimizer
optimizer = dict(type='SGD', lr=0.1, momentum=0.9, weight_decay=0.0001)
# learning policy
param_scheduler = dict(
    type='MultiStepLR', by_epoch=True, milestones=[100, 150], gamma=0.1)

# train, val, test setting
train_cfg = dict(by_epoch=True, max_epochs=200)
val_cfg = dict(interval=1)  # validate every epoch
test_cfg = dict()
