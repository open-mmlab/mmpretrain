_base_ = './efficientformer-l1_8xb128_in1k.py'

model = dict(backbone=dict(arch='l7'), head=dict(in_channels=768))
