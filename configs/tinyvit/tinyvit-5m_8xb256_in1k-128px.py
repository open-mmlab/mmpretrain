_base_ = [
    '../_base_/datasets/imagenet_bs32_pil_bicubic_128.py',
    '../_base_/schedules/imagenet_bs1024_adamw_swin.py',
    '../_base_/default_runtime.py',
    '../_base_/models/tinyvit/tinyvit-5m_custom_128.py',
]
