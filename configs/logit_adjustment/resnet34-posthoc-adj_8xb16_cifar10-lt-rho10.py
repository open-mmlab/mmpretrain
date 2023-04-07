_base_ = ['resnet34_8xb16_cifar10-lt-rho10.py']

# model settings
model = dict(
    head=dict(
        type='LogitAdjustLinearClsHead',
        num_classes=10,
        in_channels=512,
        enable_posthoc_adjustment=True,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
    ))
