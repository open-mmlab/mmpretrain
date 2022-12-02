_base_ = [
    '../_base_/models/replknet-XL_in1k.py',
    '../_base_/datasets/imagenet_bs8_pil_bicubic_320.py',
    '../_base_/schedules/imagenet_bs256_coslr.py',
    '../_base_/default_runtime.py'
]

# schedule settings
param_scheduler = dict(
    type='CosineAnnealingLR', T_max=300, by_epoch=True, begin=0, end=300)

train_cfg = dict(by_epoch=True, max_epochs=300)
