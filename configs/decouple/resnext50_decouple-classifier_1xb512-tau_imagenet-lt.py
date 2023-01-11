_base_ = [
    '../_base_/models/resnext50_32x4d.py',
    '../_base_/datasets/imagenet-lt_bs512_decouple.py',
    '../_base_/schedules/imagenet-lt_bs512_coslr_90e.py',
    '../_base_/default_runtime.py'
]
