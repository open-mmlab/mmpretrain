_base_ = [
    '../_base_/models/poolformer/poolformer_s12.py',
    '../_base_/datasets/imagenet_bs128_poolformer_small_224.py',
    '../_base_/schedules/imagenet_bs1024_adamw_swin.py',
    '../_base_/default_runtime.py',
]

optimizer = dict(lr=4e-3)
