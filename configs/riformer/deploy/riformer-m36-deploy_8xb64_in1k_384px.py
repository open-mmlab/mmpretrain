_base_ = '../riformer-m36_8xb64_in1k_384px.py'

model = dict(backbone=dict(deploy=True))
