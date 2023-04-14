_base_ = './efficientformer-l1_8xb128_in1k.py'

model = dict(backbone=dict(arch='l3'), head=dict(in_channels=512))
