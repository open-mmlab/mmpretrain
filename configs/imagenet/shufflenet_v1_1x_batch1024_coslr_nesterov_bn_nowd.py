_base_ = [
    '../_base_/models/shufflenet_v2_1x.py',
    '../_base_/datasets/imagenet_bs64.py',
    '../_base_/schedules/imagenet_bs1024_coslr_nesterov_bn_nowd.py',
    '../_base_/default_runtime.py'
]
