_base_ = './repvgg-B3_8xb32_in1k.py'

model = dict(backbone=dict(arch='B3g4'))
