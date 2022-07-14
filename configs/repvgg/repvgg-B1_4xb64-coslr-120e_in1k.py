_base_ = './repvgg-A0_4xb64-coslr-120e_in1k.py'

model = dict(backbone=dict(arch='B1'), head=dict(in_channels=2048))

# NOTE: `auto_scale_lr` is for automatically scaling LR,
# USER SHOULD NOT CHANGE ITS VALUES.
# base_batch_size = (4 GPUs) x (64 samples per GPU)
auto_scale_lr = dict(base_batch_size=256)
