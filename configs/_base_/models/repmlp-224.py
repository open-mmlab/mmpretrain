# model settings
model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='RepMLPNet',
        channels=(96, 192, 384, 768), 
        hs=(56,28,14,7), 
        ws=(56,28,14,7),
        num_blocks=(2,2,12,2),
        reparam_conv_k=(1, 3), 
        sharesets_nums=(1,4,32,128),
        deploy=False),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='LinearClsHead',
        num_classes=1000,
        in_channels=768,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        topk=(1, 5),
    ))
