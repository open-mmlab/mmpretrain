_base_ = '../levit-384_8xb256_in1k.py'

model = dict(backbone=dict(deploy=True), head=dict(deploy=True))
