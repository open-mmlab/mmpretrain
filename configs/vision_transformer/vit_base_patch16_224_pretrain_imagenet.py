# Refer to ClassyVision. In this implementation, label smoothing is not used.
_base_ = [
    '../_base_/models/vit_base_patch16_224_pretrain.py',
    '../_base_/datasets/imagenet_bs64_pil_resize.py',
    '../_base_/schedules/imagenet_bs4096.py', '../_base_/default_runtime.py'
]

# 64*(8*8)=4096
