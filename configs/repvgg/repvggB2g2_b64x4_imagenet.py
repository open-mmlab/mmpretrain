_base_ = './repvggA0_b64x4_imagenet.py'

model = dict(backbone=dict(arch='B2g2'), head=dict(in_channels=2560))
