_base_ = [
    '../_base_/models/wide-resnet50.py',
    '../_base_/datasets/imagenet_bs32_pil_bicubic.py',
    '../_base_/schedules/imagenet_bs256.py', '../_base_/default_runtime.py'
]
