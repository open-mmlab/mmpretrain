_base_ = '../repvggB1g4_b64x4_imagenet_120e_coslr.py'

model = dict(backbone=dict(deploy=True))
