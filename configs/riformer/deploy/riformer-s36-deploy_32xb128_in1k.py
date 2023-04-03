_base_ = '../riformer-s36_32xb128_in1k.py'

model = dict(backbone=dict(deploy=True))
