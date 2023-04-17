_base_ = '../riformer-s24_8xb128_in1k.py'

model = dict(backbone=dict(deploy=True))
