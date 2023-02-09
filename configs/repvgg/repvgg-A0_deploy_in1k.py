_base_ = './repvgg-A0_8xb32_in1k.py'

model = dict(backbone=dict(deploy=True))
