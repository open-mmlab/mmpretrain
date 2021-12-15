_base_ = ['regnety_400mf_8xb128_in1k.py']

# model settings
model = dict(
    backbone=dict(arch='regnety_4.0gf'),
    head=dict(in_channels=1088),
)

data = dict(samples_per_gpu=64, )
