_base_ = [
    '../_base_/models/deit3/deit3-base-p16-384.py',
    '../_base_/datasets/imagenet_bs64_deit3_384.py',
    '../_base_/schedules/imagenet_bs4096_AdamW.py',
    '../_base_/default_runtime.py'
]

# dataset setting
train_dataloader = dict(batch_size=32)

# schedule settings
optim_wrapper = dict(optimizer=dict(lr=1e-5, weight_decay=0.1))

# NOTE: `auto_scale_lr` is for automatically scaling LR
# based on the actual training batch size.
# base_batch_size = (64 GPUs) x (32 samples per GPU)
auto_scale_lr = dict(base_batch_size=2048)
