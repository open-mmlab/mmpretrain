_base_ = ['./regnetx-400mf_8xb128_in1k.py']

# model settings
model = dict(
    backbone=dict(arch='regnety_400mf'),
    head=dict(in_channels=440),
)
