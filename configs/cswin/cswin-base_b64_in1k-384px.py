_base_ = [
    '../_base_/models/cswin/base_384.py',
    '../_base_/datasets/imagenet_bs64_cswin_384.py',
    '../_base_/schedules/imagenet_bs1024_adamw_cswin.py',
    '../_base_/default_runtime.py'
]

# schedule settings
optim_wrapper = dict(clip_grad=dict(max_norm=5.0))
