_base_ = '../riformer-s36_8xb64_in1k-384px.py'

model = dict(backbone=dict(deploy=True))
