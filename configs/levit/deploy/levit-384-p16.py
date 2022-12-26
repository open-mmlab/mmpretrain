_base_ = ['../levit-384-p16.py']

model = dict(backbone=dict(deploy=True), head=dict(deploy=True))
