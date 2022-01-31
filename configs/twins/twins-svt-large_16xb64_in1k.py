_base_ = ['twins-svt-base_8xb128_in1k.py']

data = dict(samples_per_gpu=64)

model = dict(backbone=dict(arch='large'), head=dict(in_channels=1024))
