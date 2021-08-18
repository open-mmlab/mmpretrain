_base_ = '../repvggB1g2_b64x4_imagenet_120e_coslr.py'

model = dict(backbone=dict(deploy=True))
