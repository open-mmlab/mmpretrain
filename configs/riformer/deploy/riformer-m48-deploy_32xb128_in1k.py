_base_ = '../riformer-m48_32xb128_in1k.py'

model = dict(backbone=dict(deploy=True))
