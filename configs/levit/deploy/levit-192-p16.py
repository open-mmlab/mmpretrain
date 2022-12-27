_base_ = ['../levit-192-p16.py']

model = dict(
    backbone=dict(deploy=True), head=dict(deploy=True, distillation=True))
