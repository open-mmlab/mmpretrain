_base_ = '../repvggA0_b64x4_imagenet.py'

model = dict(backbone=dict(deploy=True))
