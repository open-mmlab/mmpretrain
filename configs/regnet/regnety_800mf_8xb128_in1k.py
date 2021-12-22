_base_ = ['./regnetx-400mf_8xb128_in1k.py']

# model settings
model = dict(
    backbone=dict(arch='regnety_800mf'),
    head=dict(in_channels=768),
)
