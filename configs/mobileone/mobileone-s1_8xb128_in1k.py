_base_ = [
    '../_base_/models/mobileone/mobileone_s1.py',
    '../_base_/datasets/imagenet_bs32_pil_resize.py',
    '../_base_/schedules/imagenet_bs256_coslr.py',
    '../_base_/default_runtime.py'
]

# dataset settings
train_dataloader = dict(batch_size=128, num_workers=12)
val_dataloader = dict(batch_size=128, num_workers=12)
test_dataloader = dict(batch_size=128, num_workers=12)
