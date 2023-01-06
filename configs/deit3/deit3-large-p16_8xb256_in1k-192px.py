_base_ = [
    '../_base_/models/deit3/deit3-large-p16-224.py',
    '../_base_/datasets/imagenet_bs256_deit3.py',
    '../_base_/schedules/imagenet_bs2048_deit3.py',
    '../_base_/default_runtime.py'
]

# model settings
model = dict(backbone=dict(img_size=192))
