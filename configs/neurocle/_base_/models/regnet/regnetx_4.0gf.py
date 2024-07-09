# model settings
model = dict(
    type='ImageClassifier',
    pretrained='https://download.openmmlab.com/mmclassification/v0/regnet/regnetx-4.0gf_8xb64_in1k_20211213-efed675c.pth', 
    backbone=dict(type='RegNet', arch='regnetx_4.0gf'),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='LinearClsHead',
        num_classes=1000,
        in_channels=1360,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        topk=(1, 5),
    ))
