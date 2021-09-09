_base_ = '../repvggA0_4xb64-coslr-120e_in-1k.py'

model = dict(backbone=dict(deploy=True))
