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
