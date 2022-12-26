_base_ = ['../levit-128s-p16.py']

model = dict(backbone=dict(deploy=True), head=dict(deploy=True))
