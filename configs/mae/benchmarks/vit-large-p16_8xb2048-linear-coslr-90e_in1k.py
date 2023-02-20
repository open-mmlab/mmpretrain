_base_ = 'vit-base-p16_8xb2048-linear-coslr-90e_in1k.py'
# model settings
model = dict(
    backbone=dict(type='VisionTransformer', arch='large', frozen_stages=24),
    neck=dict(type='ClsBatchNormNeck', input_features=1024),
    head=dict(in_channels=1024))
