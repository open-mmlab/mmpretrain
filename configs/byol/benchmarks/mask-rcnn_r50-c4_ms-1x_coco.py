_base_ = 'mmdet::mask_rcnn/mask-rcnn_r50-caffe-c4_1x_coco.py'
# https://github.com/open-mmlab/mmdetection/blob/dev-3.x/configs/mask_rcnn/mask-rcnn_r50-caffe-c4_1x_coco.py

data_preprocessor = dict(
    type='DetDataPreprocessor',
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    bgr_to_rgb=True,
    pad_mask=True,
    pad_size_divisor=32)

norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    data_preprocessor=data_preprocessor,
    backbone=dict(
        frozen_stages=-1,
        norm_cfg=norm_cfg,
        norm_eval=False,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')),
    roi_head=dict(
        shared_head=dict(
            type='ResLayerExtraNorm',
            norm_cfg=norm_cfg,
            norm_eval=False,
            style='pytorch')))

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(
        type='RandomChoiceResize',
        scales=[(1333, 640), (1333, 672), (1333, 704), (1333, 736),
                (1333, 768), (1333, 800)],
        keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackDetInputs')
]

train_dataloader = dict(dataset=dict(pipeline=train_pipeline))

train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=12, val_interval=1)

custom_imports = dict(
    imports=['mmpretrain.models.utils.res_layer_extra_norm'],
    allow_failed_imports=False)
