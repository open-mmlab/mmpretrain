_base_ = [
    '../_base_/models/davit/davit-small.py',
    '../_base_/datasets/imagenet_bs256_davit_224.py',
    '../_base_/schedules/imagenet_bs1024_adamw_swin.py',
    '../_base_/default_runtime.py'
]

# data settings
train_dataloader = dict(batch_size=256)
