_base_ = ['twins-svt-base_8xb128_in1k.py']

# model settings
model = dict(backbone=dict(arch='small'), head=dict(in_channels=512))

# NOTE: `auto_scale_lr` is for automatically scaling LR,
# USER SHOULD NOT CHANGE ITS VALUES.
# base_batch_size = (8 GPUs) x (128 samples per GPU)
auto_scale_lr = dict(base_batch_size=1024)
