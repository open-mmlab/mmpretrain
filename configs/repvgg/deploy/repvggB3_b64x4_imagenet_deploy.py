_base_ = '../repvggB3_64x4_imagenet.py'

model = dict(backbone=dict(deploy=True))
