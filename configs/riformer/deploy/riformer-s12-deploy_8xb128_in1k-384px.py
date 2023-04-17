_base_ = '../riformer-s12_8xb128_in1k-384px.py'

model = dict(backbone=dict(deploy=True))
