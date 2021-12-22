_base_ = './deit-small_pt-4xb256_in1k.py'

# model settings
model = dict(
    backbone=dict(type='DistilledVisionTransformer', arch='deit-base'),
    head=dict(type='DeiTClsHead', in_channels=768),
)

# data settings
data = dict(samples_per_gpu=64, workers_per_gpu=5)
