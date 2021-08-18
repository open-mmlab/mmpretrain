_base_ = '../repvggB3g2_b64x4_imagenet.py'

model = dict(backbone=dict(deploy=True))
