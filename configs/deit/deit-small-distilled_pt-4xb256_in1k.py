_base_ = './deit-small_pt-4xb256_in1k.py'

# model settings
model = dict(
    backbone=dict(type='DistilledVisionTransformer', arch='deit-small'),
    head=dict(type='DeiTClsHead', in_channels=384),
)
