_base_ = './repvggA0_b64x4_imagenet_120e_coslr.py'

model = dict(backbone=dict(arch='B1g4'), head=dict(in_channels=2048))
