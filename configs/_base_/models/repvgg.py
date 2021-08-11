# model settings
model = dict(
    type='ImageClassifier',
    backbone=dict(
            type='RepVGG',
            num_blocks=[2, 4, 14, 1],
            width_multiplier=[0.75, 0.75, 0.75, 2.5],
            override_groups_map=dict(),
            deploy=False),
    neck=None,
    head=dict(
            type='LinearClsHead',
            num_classes=1000,
            in_channels=1280,
            loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
            topk=(1, 5),
    ))