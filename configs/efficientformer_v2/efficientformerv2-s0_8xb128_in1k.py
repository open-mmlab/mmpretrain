model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='EfficientFormerV2',
        arch='s0',
        drop_path_rate=0.,
        init_cfg=[
            dict(
                type='TruncNormal',
                layer=['Conv2d', 'Linear'],
                std=.02,
                bias=0.),
            dict(type='Constant', layer=['GroupNorm'], val=1., bias=0.),
            dict(type='Constant', layer=['LayerScale'], val=1e-5)
        ]),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='EfficientFormerClsHead', in_channels=176, num_classes=1000))
