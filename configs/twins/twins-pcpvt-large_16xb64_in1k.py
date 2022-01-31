_base_ = ['twins-pcpvt-base_8xb128_in1k.py']

model = dict(backbone=dict(arch='large'), head=dict(in_channels=512))

data = dict(samples_per_gpu=64)
