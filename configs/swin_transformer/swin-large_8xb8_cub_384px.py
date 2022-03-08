_base_ = [
    '../_base_/models/swin_transformer/large_384.py',
    '../_base_/datasets/cub_bs8_384.py', '../_base_/schedules/cub_bs64.py',
    '../_base_/default_runtime.py'
]

# model settings
checkpoint = 'https://download.openmmlab.com/mmclassification/v0/swin-transformer/convert/swin-large_3rdparty_in21k-384px.pth'  # noqa
model = dict(
    type='ImageClassifier',
    backbone=dict(
        init_cfg=dict(
            type='Pretrained', checkpoint=checkpoint, prefix='backbone')),
    head=dict(num_classes=200, ))

paramwise_cfg = dict(
    norm_decay_mult=0.0,
    bias_decay_mult=0.0,
    custom_keys={
        '.absolute_pos_embed': dict(decay_mult=0.0),
        '.relative_position_bias_table': dict(decay_mult=0.0)
    })

optimizer = dict(
    _delete_=True,
    type='AdamW',
    lr=5e-6,
    weight_decay=0.0005,
    eps=1e-8,
    betas=(0.9, 0.999),
    paramwise_cfg=paramwise_cfg)
optimizer_config = dict(grad_clip=dict(max_norm=5.0), _delete_=True)

log_config = dict(interval=20)  # log every 20 intervals

checkpoint_config = dict(
    interval=1, max_keep_ckpts=3)  # save last three checkpoints
