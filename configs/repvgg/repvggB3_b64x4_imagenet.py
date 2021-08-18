_base_ = './repvggA0_b64x4_imagenet.py'

model = dict(backbone=dict(arch='B3'), head=dict(in_channels=2560))
