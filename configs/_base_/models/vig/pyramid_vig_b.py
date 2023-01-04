# model settings
model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='pyramid_vig',
        k=9,
        conv='mr',
        act='gelu',
        bias=True,
        dropout=0.0,
        epsilon=0.2,
        use_stochastic=False,
        drop_path=0,
        blocks=[2, 2, 18, 2],
        channels=[128, 256, 512, 1024],
        n_classes=1000),
    head=dict(
        type='ClsHead',
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        topk=(1, 5),
    ))
