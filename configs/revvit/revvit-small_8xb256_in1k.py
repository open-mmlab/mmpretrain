_base_ = [
    '../_base_/models/revvit/revvit-small.py',
    '../_base_/datasets/imagenet_bs128_revvit_224.py',
    '../_base_/schedules/imagenet_bs1024_adamw_revvit.py',
    '../_base_/default_runtime.py'
]
