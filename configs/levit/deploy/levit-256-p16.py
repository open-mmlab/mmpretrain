_base_ = ['../levit-256-p16.py']

model = dict(backbone=dict(deploy=True), head=dict(deploy=True))
