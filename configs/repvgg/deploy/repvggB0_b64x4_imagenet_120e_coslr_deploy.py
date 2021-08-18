_base_ = '../repvggB0_b64x4_imagenet_120e_coslr.py'

model = dict(backbone=dict(deploy=True))
