_base_ = [
    '../_base_/models/mobilenet_v3_small_imagenet.py',
    '../_base_/datasets/imagenet_bs32_pil_resize.py',
    '../_base_/schedules/imagenet_bs256_epochstep.py',
    '../_base_/default_runtime.py'
]

optimizer = dict(type='SGD', lr=0.18, momentum=0.9, weight_decay=0.00004)
