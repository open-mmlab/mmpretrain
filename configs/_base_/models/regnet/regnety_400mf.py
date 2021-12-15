# model settings
model = dict(
    type='ImageClassifier',
    backbone=dict(type='RegNet', arch='regnety_400mf'),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='LinearClsHead',
        num_classes=1000,
        in_channels=440,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        topk=(1, 5),
    ))
'''
for different regnety-xxx, the head should have different in_channels
here are the map:

regnety_400mf : 440
regnety_600mf : 608
regnety_800mf : 768
regnety_1.6gf : 888
regnety_3.2gf : 1512
regnety_4.0gf : 1088
regnety_6.4gf : 1296
regnety_8.0gf : 2016
regnety_12gf : 2240
'''
