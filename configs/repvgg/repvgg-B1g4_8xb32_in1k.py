_base_ = './repvgg-A0_8xb32_in1k.py'

model = dict(backbone=dict(arch='B1g4'), head=dict(in_channels=2048))
