_base_ = '../repvgg-B2_4xb64-coslr-120e_in1k.py'

model = dict(backbone=dict(deploy=True))
