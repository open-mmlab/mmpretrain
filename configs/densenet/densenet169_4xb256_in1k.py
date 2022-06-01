_base_ = [
    '../_base_/models/densenet/densenet169.py',
    '../_base_/datasets/imagenet_bs64.py',
    '../_base_/schedules/imagenet_bs256.py',
    '../_base_/default_runtime.py',
]

# dataset settings
train_dataloader = dict(batch_size=256)

# schedule settings
train_cfg = dict(by_epoch=True, max_epochs=90)
