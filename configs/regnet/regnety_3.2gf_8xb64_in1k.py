_base_ = ['./regnetx-400mf_8xb128_in1k.py']

# model settings
model = dict(
    backbone=dict(arch='regnety_3.2gf'),
    head=dict(in_channels=1512),
)

data = dict(samples_per_gpu=64, )
