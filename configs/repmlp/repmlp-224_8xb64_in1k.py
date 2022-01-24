_base_ = [
    '../_base_/models/repmlp-224.py',
    '../_base_/datasets/imagenet_bs64_mixer_224.py',
    '../_base_/schedules/imagenet_bs4096_AdamW.py',
    '../_base_/default_runtime.py'
]