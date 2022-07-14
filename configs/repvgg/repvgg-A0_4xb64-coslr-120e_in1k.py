_base_ = [
    '../_base_/models/repvgg-A0_in1k.py',
    '../_base_/datasets/imagenet_bs64_pil_resize.py',
    '../_base_/schedules/imagenet_bs256_coslr.py',
    '../_base_/default_runtime.py'
]

# schedule settings
param_scheduler = dict(
    type='CosineAnnealingLR', T_max=120, by_epoch=True, begin=0, end=120)

train_cfg = dict(by_epoch=True, max_epochs=120)

# NOTE: `auto_scale_lr` is for automatically scaling LR,
# USER SHOULD NOT CHANGE ITS VALUES.
# base_batch_size = (4 GPUs) x (64 samples per GPU)
auto_scale_lr = dict(base_batch_size=256)
