_base_ = './deit-small_pt-4xb256_in1k.py'

# model settings
model = dict(
    backbone=dict(type='VisionTransformer', arch='deit-tiny'),
    head=dict(type='VisionTransformerClsHead', in_channels=192),
)
