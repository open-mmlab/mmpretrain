_base_ = ['../levit-256-p16_4xb256_autoaug-mixup-lbs-coslr-1000e_in1k.py']

model = dict(
    backbone=dict(deploy=True), head=dict(deploy=True, distillation=True))
