model_cfg = dict(
    type='ImageClassifier',
    backbone=dict(
        type='vig',
        k=9,
        n_classes=1000,
        n_blocks=12,
        epsilon=0.2,
        use_stochastic=False,
        drop_path=0,
        use_dilation=True,
        channels=192,
        dropout=0),
    head=dict(
        type='ClsHead',
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        topk=(1, 5),
    ))
