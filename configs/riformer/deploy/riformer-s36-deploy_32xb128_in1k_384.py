_base_ = '../riformer-s36_32xb128_in1k_384.py'

model = dict(backbone=dict(deploy=True))
