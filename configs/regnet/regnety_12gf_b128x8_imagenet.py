_base_ = [
    '../_base_/models/regnet/regnety_12gf.py',
    '../_base_/datasets/imagenet_bs128.py',
    '../_base_/schedules/imagenet_bs1024_coslr_warmup.py',
    '../_base_/default_runtime.py'
]
