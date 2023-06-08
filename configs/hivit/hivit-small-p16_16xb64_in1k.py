_base_ = [
    '../_base_/models/hivit/small_224.py',
    '../_base_/datasets/imagenet_bs64_hivit_224.py',
    '../_base_/schedules/imagenet_bs1024_adamw_hivit.py',
    '../_base_/default_runtime.py'
]

# schedule settings
optim_wrapper = dict(clip_grad=dict(max_norm=5.0))
