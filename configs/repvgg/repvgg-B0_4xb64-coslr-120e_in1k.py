_base_ = './repvgg-A0_4xb64-coslr-120e_in1k.py'

model = dict(backbone=dict(arch='B0'), head=dict(in_channels=1280))
