_base_ = [
    '../_base_/datasets/inshop_bs8_448.py',
    '../_base_/schedules/cub_bs64.py',
    '../_base_/default_runtime.py',
]

train_dataloader = dict(batch_size=32, num_workers=8)
gallery_dataloader = dict(batch_size=32, num_workers=8)
val_dataloader = dict(batch_size=32, num_workers=8)
test_dataloader = dict(batch_size=32, num_workers=8)

pretrained = 'https://download.openmmlab.com/mmclassification/v0/resnet/resnet50_3rdparty-mill_in21k_20220331-faac000b.pth'  # noqa
model = dict(
    type='ImageToImageRetriever',
    image_encoder=[
        dict(
            type='ResNet',
            depth=50,
            num_stages=4,
            out_indices=(3, ),
            style='pytorch',
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
    prototype=gallery_dataloader)

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

auto_scale_lr = dict(enable=True, base_batch_size=64)

custom_hooks = [
    dict(type='PrepareProtoBeforeValLoopHook')
]
