_base_ = ['twins-pcpvt-base_8xb128_in1k.py']

# model settings
model = dict(backbone=dict(arch='large'), head=dict(in_channels=512))

# dataset settings
train_dataloader = dict(batch_size=128)

# NOTE: `auto_scale_lr` is for automatically scaling LR,
# USER SHOULD NOT CHANGE ITS VALUES.
# base_batch_size = (16 GPUs) x (64 samples per GPU)
auto_scale_lr = dict(base_batch_size=1024)
