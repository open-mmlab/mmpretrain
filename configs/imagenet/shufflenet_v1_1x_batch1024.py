_base_ = [
    '../_base_/models/shufflenet_v1_1x.py',
    '../_base_/datasets/imagenet_bs64.py',
    '../_base_/schedules/imagenet_bs1024_linearlr.py',
    '../_base_/default_runtime.py'
]
