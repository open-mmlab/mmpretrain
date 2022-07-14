_base_ = ['./repmlp-base_8xb64_in1k.py']

model = dict(backbone=dict(deploy=True))

# NOTE: `auto_scale_lr` is for automatically scaling LR,
# USER SHOULD NOT CHANGE ITS VALUES.
# base_batch_size = (8 GPUs) x (64 samples per GPU)
auto_scale_lr = dict(base_batch_size=512)
