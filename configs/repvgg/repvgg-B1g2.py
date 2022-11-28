_base_ = './repvgg-A0.py'

model = dict(backbone=dict(arch='B1g2'), head=dict(in_channels=2048))
