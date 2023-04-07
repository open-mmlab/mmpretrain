_base_ = '../riformer-m48_8xb64_in1k.py'

model = dict(backbone=dict(deploy=True))
