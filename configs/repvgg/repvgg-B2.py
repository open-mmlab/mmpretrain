_base_ = './repvgg-A0.py'

model = dict(backbone=dict(arch='B2'), head=dict(in_channels=2560))
