_base_ = [
    '../_base_/models/conformer/small_224.py',
    '../_base_/datasets/imagenet_bs64_swin_224.py',
    '../_base_/schedules/imagenet_bs1024_adamw_conformer.py',
    '../_base_/default_runtime.py'
]

data = dict(
    samples_per_gpu=128,
    workers_per_gpu=8,
    val=dict(ann_file=None),
    test=dict(ann_file=None),
)

evaluation = dict(interval=1, metric='accuracy')
