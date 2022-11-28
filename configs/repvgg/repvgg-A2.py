_base_ = './repvgg-A0.py'

model = dict(backbone=dict(arch='A2'), head=dict(in_channels=1408))
