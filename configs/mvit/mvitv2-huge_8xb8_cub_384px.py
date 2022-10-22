_base_ = [
    '../_base_/models/mvit/mvitv2-huge.py',
    '../_base_/datasets/cub_bs8_384.py', '../_base_/schedules/cub_bs64.py',
    '../_base_/default_runtime.py'
]

# model settings
# todo: change to url
checkpoint = 'MViTv2_H_in21k.pth'  # noqa
model = dict(
    type='ImageClassifier',
    backbone=dict(
        with_cp=True,
        dim_mul_in_attention=True,
        init_cfg=dict(
            type='Pretrained', checkpoint=checkpoint, prefix='backbone')),
    head=dict(num_classes=200, ))

# schedule settings
optim_wrapper = dict(
    optimizer=dict(
        _delete_=True,
        type='AdamW',
        lr=5e-6,
        weight_decay=0.0005,
        eps=1e-8,
        betas=(0.9, 0.999)),
    paramwise_cfg=dict(custom_keys={
        '.cls_token': dict(decay_mult=0.0),
        '.pos_embed': dict(decay_mult=0.0)
    }),
    clip_grad=dict(max_norm=5.0),
)

default_hooks = dict(
    # log every 20 intervals
    logger=dict(type='LoggerHook', interval=20),
    # save last three checkpoints
    checkpoint=dict(type='CheckpointHook', interval=1, max_keep_ckpts=3))
