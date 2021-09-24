_base_ = [
    '../_base_/models/repvgg-B3_lbs-mixup_in1k.py',
    '../_base_/datasets/imagenet_bs64_pil_resize.py',
    '../_base_/schedules/imagenet_bs256_200e_coslr_warmup.py',
    '../_base_/default_runtime.py'
]
