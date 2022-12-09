# model settings
model_cfg = dict(
    type='ImageClassifier',
    backbone=dict(
        type='pyramid_vig',
        k=9,
        dropout=0,
        use_dilation=True,
        epsilon=0.2,
        use_stochastic=False,
        drop_path=0,
        blocks=[2, 2, 16, 2],
        channels=[96, 192, 384, 768],
        n_classes=1000,
        emb_dims=1024),
    head=dict(
        type='ClsHead',
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        topk=(1, 5),
    ))
