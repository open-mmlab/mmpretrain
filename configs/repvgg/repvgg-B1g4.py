_base_ = './repvgg-A0.py'

model = dict(backbone=dict(arch='B1g4'), head=dict(in_channels=2048))
