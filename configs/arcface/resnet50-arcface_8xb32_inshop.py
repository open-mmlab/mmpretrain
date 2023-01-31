_base_ = [
    '../_base_/datasets/inshop_bs32_448.py',
    '../_base_/schedules/cub_bs64.py',
    '../_base_/default_runtime.py',
]

pretrained = 'https://download.openmmlab.com/mmclassification/v0/resnet/resnet50_3rdparty-mill_in21k_20220331-faac000b.pth'  # noqa
model = dict(
    type='ImageToImageRetriever',
    image_encoder=[
        dict(
            type='ResNet',
            depth=50,
            init_cfg=dict(
                type='Pretrained', checkpoint=pretrained, prefix='backbone')),
        dict(type='GlobalAveragePooling'),
    ],
    head=dict(
        type='ArcFaceClsHead',
        num_classes=3997,
        in_channels=2048,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        init_cfg=None),
    prototype={{_base_.gallery_dataloader}})

# runtime settings
default_hooks = dict(
    # log every 20 intervals
    logger=dict(type='LoggerHook', interval=20),
    # save last three checkpoints
    checkpoint=dict(
        type='CheckpointHook',
        save_best='auto',
        interval=1,
        max_keep_ckpts=3,
        rule='greater'))

# optimizer
optim_wrapper = dict(
    optimizer=dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0005, nesterov=True))

auto_scale_lr = dict(enable=True, base_batch_size=256)

custom_hooks = [dict(type='PrepareProtoBeforeValLoopHook')]
