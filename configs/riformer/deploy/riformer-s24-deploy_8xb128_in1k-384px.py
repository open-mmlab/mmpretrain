_base_ = '../riformer-s24_8xb128_in1k-384px.py'

model = dict(backbone=dict(deploy=True))
