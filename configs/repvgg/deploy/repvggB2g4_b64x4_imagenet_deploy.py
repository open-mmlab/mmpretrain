_base_ = '../repvggB2g4_b64x4_imagenet.py'

model = dict(backbone=dict(deploy=True))
