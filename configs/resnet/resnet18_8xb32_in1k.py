_base_ = [
    '../_base_/models/resnet18.py', '../_base_/datasets/imagenet_bs32.py',
    '../_base_/schedules/imagenet_bs256.py', '../_base_/default_runtime.py'
]

# schedule settings
param_scheduler = dict(
    type='CosineAnnealingLR',
    T_max=120,
    by_epoch=True,
    begin=0,
    end=120,
    convert_to_iter_based=True)

train_cfg = dict(by_epoch=True, max_epochs=120)
