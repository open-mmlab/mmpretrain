_base_ = '../riformer-m36_8xb128_in1k.py'

model = dict(backbone=dict(deploy=True))
