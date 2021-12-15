_base_ = ['regnety_400mf_8xb128_in1k.py']

# model settings
model = dict(
    backbone=dict(arch='regnety_6.4gf'),
    head=dict(in_channels=1296),
)

data = dict(samples_per_gpu=64, )
