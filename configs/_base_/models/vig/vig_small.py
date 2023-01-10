model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='Vig',
        arch='small',
        k=9,
        act='GELU',
        norm='batch',
        bias=True,
        use_dilation=True,
        epsilon=0.2,
        use_stochastic=False,
        conv='mr',
        drop_path=0.0,
        dropout=0.0,
        n_classes=1000,
        relative_pos=False),
    neck=None,
    head=dict(
        type='ClsHead',
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        topk=(1, 5),
    ))
