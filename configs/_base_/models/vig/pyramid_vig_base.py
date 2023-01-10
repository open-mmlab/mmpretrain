# model settings
model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='PyramidVig',
        arch='base',
        k=9,
        conv='mr',
        act='GELU',
        norm='batch',
        bias=True,
        dropout=0.0,
        epsilon=0.2,
        use_stochastic=False,
        drop_path=0,
        n_classes=1000),
    head=dict(
        type='ClsHead',
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        topk=(1, 5),
    ))
