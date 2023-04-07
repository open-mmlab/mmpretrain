_base_ = ['resnet34_8xb16_cifar100-lt-rho10.py']

# model settings
model = dict(
    head=dict(
        num_classes=100,
        in_channels=512,
        type='LogitAdjustLinearClsHead',
        enable_loss_adjustment=True,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
    ))
