# model settings
model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='PyramidVig',
        arch='base',
        k=9,
        act_cfg=dict(type='GELU'),
        norm_cfg='batch',
        graph_conv_type='mr',
        graph_conv_bias=True,
        epsilon=0.2,
        use_stochastic=False,
        drop_path=0.,
        dropout=0.,
        n_classes=1000,
        norm_eval=False,
        frozen_stages=0,
        init_cfg=None),
    head=dict(
        type='ClsHead',
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        topk=(1, 5),
    ))
