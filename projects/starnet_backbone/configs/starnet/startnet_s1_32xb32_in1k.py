_base_ = [
    '../_base_/models/starnet/starnet_s1.py',
    '../_base_/datasets/imagenet_bs32.py',
    '../_base_/schedules/imagenet_bs256.py', '../_base_/default_runtime.py'
]
train_dataloader = dict(batch_size=24)
val_dataloader = dict(batch_size=24)
