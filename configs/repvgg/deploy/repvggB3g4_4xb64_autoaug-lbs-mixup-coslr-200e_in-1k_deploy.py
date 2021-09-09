_base_ = '../repvggB3g4_4xb64_autoaug-lbs-mixup-coslr-200e_in-1k.py'

model = dict(backbone=dict(deploy=True))
