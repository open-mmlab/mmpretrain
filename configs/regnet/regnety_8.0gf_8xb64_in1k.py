_base_ = ['./regnetx-400mf_8xb128_in1k.py']

# model settings
model = dict(
    backbone=dict(arch='regnety_8.0gf'),
    head=dict(in_channels=2016),
)

data = dict(samples_per_gpu=64, )
