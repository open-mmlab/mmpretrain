_base_ = '../repvggB3g4_64x4_imagenet.py'

model = dict(backbone=dict(deploy=True))
