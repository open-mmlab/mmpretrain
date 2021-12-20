_base_ = './deit-small_pt-4xb256_in1k.py'

# model settings
model = dict(
    backbone=dict(type='DistilledVisionTransformer', arch='deit-tiny'),
    head=dict(type='DeiTClsHead', in_channels=192),
)
