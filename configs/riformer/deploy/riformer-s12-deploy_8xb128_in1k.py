_base_ = '../riformer-s12_8xb128_in1k.py'

model = dict(backbone=dict(deploy=True))
