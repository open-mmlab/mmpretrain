_base_ = ['./edgenext-xxsmall_8xb256_in1k.py']

# Model settings
model = dict(backbone=dict(arch='xsmall'), head=dict(in_channels=192))
