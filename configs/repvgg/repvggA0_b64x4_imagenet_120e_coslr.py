_base_ = [
    '../_base_/models/repvggA0.py', '../_base_/datasets/imagenet_bs64.py',
    '../_base_/schedules/imagenet_bs256_coslr.py',
    '../_base_/default_runtime.py'
]

runner = dict(max_epochs=120)
