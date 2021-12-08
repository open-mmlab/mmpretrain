_base_ = '../repvgg-B2g4_4xb64-autoaug-lbs-mixup-coslr-200e_in1k.py'

model = dict(backbone=dict(deploy=True))
