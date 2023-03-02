_base_ = [
    'mmcls::_base_/models/clip/clip-l-336.py',
    'mmcls::_base_/datasets/imagenet_bs64_clip_336.py',
    'mmcls::_base_/schedules/imagenet_bs4096_AdamW.py',
    'mmcls::_base_/default_runtime.py'
]
