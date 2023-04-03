_base_ = '../riformer-s24_32xb128_in1k_384.py'

model = dict(backbone=dict(deploy=True))
