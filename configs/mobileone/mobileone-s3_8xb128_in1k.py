_base_ = [
    '../_base_/models/mobileone/mobileone_s3.py',
    '../_base_/datasets/imagenet_bs64_pil_resize.py',
    '../_base_/schedules/imagenet_bs256_coslr.py',
    '../_base_/default_runtime.py'
]

# dataset settings
train_dataloader = dict(batch_size=128)
val_dataloader = dict(batch_size=128)
test_dataloader = dict(batch_size=128)

# NOTE: `auto_scale_lr` is for automatically scaling LR,
# based on the actual training batch size.
auto_scale_lr = dict(base_batch_size=1024)
