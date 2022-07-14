_base_ = [
    '../_base_/models/conformer/small-p32.py',
    '../_base_/datasets/imagenet_bs64_swin_224.py',
    '../_base_/schedules/imagenet_bs1024_adamw_conformer.py',
    '../_base_/default_runtime.py'
]

train_dataloader = dict(batch_size=128)

# NOTE: `auto_scale_lr` is for automatically scaling LR,
# USER SHOULD NOT CHANGE ITS VALUES.
# base_batch_size = (8 GPUs) x (128 samples per GPU)
auto_scale_lr = dict(base_batch_size=1024)
