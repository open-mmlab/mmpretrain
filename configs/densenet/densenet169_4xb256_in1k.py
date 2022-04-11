_base_ = [
    '../_base_/models/densenet/densenet169.py',
    '../_base_/datasets/imagenet_bs64.py',
    '../_base_/schedules/imagenet_bs256.py',
    '../_base_/default_runtime.py',
]

data = dict(samples_per_gpu=256)

runner = dict(type='EpochBasedRunner', max_epochs=90)
