_base_ = [
    '../_base_/models/vit_base_patch16_224_pretrain.py',
    '../_base_/datasets/imagenet_bs32_pil_resize.py',
    '../_base_/schedules/imagenet_bs4096.py', '../_base_/default_runtime.py'
]
