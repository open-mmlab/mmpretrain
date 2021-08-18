_base_ = '../repvggB2g2_b64x4_imagenet.py'

model = dict(backbone=dict(deploy=True))
