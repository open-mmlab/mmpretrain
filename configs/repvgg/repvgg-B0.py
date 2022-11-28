_base_ = './repvgg-A0.py'

model = dict(backbone=dict(arch='B0'), head=dict(in_channels=1280))
