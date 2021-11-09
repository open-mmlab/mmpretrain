_base_ = [
    '../_base_/models/mlp_mixer_b16.py',
    '../_base_/datasets/imagenet_bs64_mixer_224.py',
    '../_base_/schedules/imagenet_bs256.py', '../_base_/default_runtime.py'
]
