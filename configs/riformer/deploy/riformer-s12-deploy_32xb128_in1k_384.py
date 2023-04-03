_base_ = '../riformer-s12_32xb128_in1k_384.py'

model = dict(backbone=dict(deploy=True))
