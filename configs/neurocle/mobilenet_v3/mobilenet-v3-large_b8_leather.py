# Refers to https://pytorch.org/blog/ml-models-torchvision-v0.9/#classification

_base_ = [
    '../_base_/models/mobilenet_v3/mobilenet_v3_large_imagenet.py',
    '../_base_/datasets/leather_bs8.py',
    '../_base_/schedules/leather.py',
    '../_base_/default_runtime.py',
]

model = dict(
    head=dict(
        num_classes={{_base_.data_preprocessor.num_classes}}
    )
)