_base_ = [
    '../_base_/models/resnet50.py',
    '../_base_/datasets/imagenet_bs32_pillow.py',
    '../../_base_/schedules/lars_coslr-90e.py',
    '../../::_base_/default_runtime.py',
]
# SimSiam linear evaluation setting
# According to SimSiam paper, this setting can also be used to evaluate
# other methods like SimCLR, MoCo, BYOL, SwAV

model = dict(backbone=dict(frozen_stages=4))

# dataset summary
train_dataloader = dict(batch_size=512)

# runtime settings
default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', interval=10, max_keep_ckpts=3))
