_base_ = [
    '../_base_/models/repmlp-base_224.py',
    '../_base_/datasets/imagenet_bs64_pil_resize.py',
    '../_base_/schedules/imagenet_bs1024_adamw_swin.py',
    '../_base_/default_runtime.py'
]

# dataset settings
test_pipeline = [
    dict(type='LoadImageFromFile'),
    # resizing to (256, 256) here, different from resizing shorter edge to 256
    dict(type='Resize', scale=(256, 256), backend='pillow'),
    dict(type='CenterCrop', crop_size=224),
    dict(type='PackInputs'),
]

val_dataloader = dict(dataset=dict(pipeline=test_pipeline))
test_dataloader = dict(dataset=dict(pipeline=test_pipeline))

# schedule settings
optim_wrapper = dict(clip_grad=dict(max_norm=5.0))

# NOTE: `auto_scale_lr` is for automatically scaling LR
# based on the actual training batch size.
# base_batch_size = (8 GPUs) x (64 samples per GPU)
auto_scale_lr = dict(base_batch_size=512)
