_base_ = [
    '../_base_/models/swin_transformer/tiny_224.py',
    '../_base_/datasets/dagm_bs8.py',
    '../_base_/schedules/dagm_swin.py',
    '../_base_/default_runtime.py'
]

# schedule settings
optim_wrapper = dict(clip_grad=dict(max_norm=5.0))
