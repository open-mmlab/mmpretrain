_base_ = './repvggB3_4xb64_autoaug-lbs-mixup-coslr-200e_in-1k.py'

model = dict(backbone=dict(arch='D2se'))
