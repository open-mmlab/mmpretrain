_base_ = [
    '../_base_/models/gcvit/gcvit_tiny.py',
    '../_base_/datasets/imagenet_bs32_pil_bicubic.py',
    '../_base_/schedules/imagenet_bs1024_adamw_conformer.py',
    '../_base_/default_runtime.py'
]