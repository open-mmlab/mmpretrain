_base_ = ['./regnetx-400mf_8xb128_in1k.py']

# model settings
model = dict(
    backbone=dict(arch='regnety_1.6gf'),
    head=dict(in_channels=888),
)
