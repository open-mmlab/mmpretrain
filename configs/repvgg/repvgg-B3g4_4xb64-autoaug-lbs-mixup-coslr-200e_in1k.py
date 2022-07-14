_base_ = './repvgg-B3_4xb64-autoaug-lbs-mixup-coslr-200e_in1k.py'

model = dict(backbone=dict(arch='B3g4'))

# NOTE: `auto_scale_lr` is for automatically scaling LR,
# USER SHOULD NOT CHANGE ITS VALUES.
# base_batch_size = (4 GPUs) x (64 samples per GPU)
auto_scale_lr = dict(base_batch_size=256)
