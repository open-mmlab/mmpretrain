_base_ = './repvgg-B3.py'

model = dict(backbone=dict(arch='B2g4'), head=dict(in_channels=2560))
