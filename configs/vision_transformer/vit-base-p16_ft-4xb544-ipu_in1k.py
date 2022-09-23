_base_ = [
    '../_base_/models/vit-base-p16.py',
    '../_base_/datasets/imagenet_bs64_pil_resize_autoaug.py',
    '../_base_/default_runtime.py'
]

# specific to vit pretrain
paramwise_cfg = dict(custom_keys={
    '.cls_token': dict(decay_mult=0.0),
    '.pos_embed': dict(decay_mult=0.0)
})

pretrained = 'https://download.openmmlab.com/mmclassification/v0/vit/pretrain/vit-base-p16_3rdparty_pt-64xb64_in1k-224_20210928-02284250.pth'  # noqa

model = dict(
    head=dict(
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0, _delete_=True), ),
    backbone=dict(
        img_size=224,
        init_cfg=dict(
            type='Pretrained',
            checkpoint=pretrained,
            _delete_=True,
            prefix='backbone')))

img_norm_cfg = dict(
    mean=[127.5, 127.5, 127.5], std=[127.5, 127.5, 127.5], to_rgb=True)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='RandomResizedCrop', size=224, backend='pillow'),
    dict(type='RandomFlip', flip_prob=0.5, direction='horizontal'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='ToTensor', keys=['gt_label']),
    dict(type='ToHalf', keys=['img']),
    dict(type='Collect', keys=['img', 'gt_label'])
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', size=(224, -1), backend='pillow'),
    dict(type='CenterCrop', crop_size=224),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='ToHalf', keys=['img']),
    dict(type='Collect', keys=['img'])
]

# change batch size
data = dict(
    samples_per_gpu=17,
    workers_per_gpu=16,
    drop_last=True,
    train=dict(pipeline=train_pipeline),
    train_dataloader=dict(mode='async'),
    val=dict(pipeline=test_pipeline, ),
    val_dataloader=dict(samples_per_gpu=4, workers_per_gpu=1),
    test=dict(pipeline=test_pipeline),
    test_dataloader=dict(samples_per_gpu=4, workers_per_gpu=1))

# remove clip-norm
optimizer_config = dict()

# optimizer
optimizer = dict(
    type='SGD',
    lr=0.08,
    weight_decay=1e-5,
    momentum=0.9,
    paramwise_cfg=paramwise_cfg,
)

# learning policy
lr_config = dict(
    policy='CosineAnnealing',
    min_lr=0,
    warmup='linear',
    warmup_iters=800,
    warmup_ratio=0.02,
)

# ipu cfg
# model partition config
ipu_model_cfg = dict(
    train_split_edges=[
        dict(layer_to_call='backbone.patch_embed', ipu_id=0),
        dict(layer_to_call='backbone.layers.3', ipu_id=1),
        dict(layer_to_call='backbone.layers.6', ipu_id=2),
        dict(layer_to_call='backbone.layers.9', ipu_id=3)
    ],
    train_ckpt_nodes=['backbone.layers.{}'.format(i) for i in range(12)])

# device config
options_cfg = dict(
    randomSeed=42,
    partialsType='half',
    train_cfg=dict(
        executionStrategy='SameAsIpu',
        Training=dict(gradientAccumulation=32),
        availableMemoryProportion=[0.3, 0.3, 0.3, 0.3],
    ),
    eval_cfg=dict(deviceIterations=1, ),
)

# add model partition config and device config to runner
runner = dict(
    type='IterBasedRunner',
    ipu_model_cfg=ipu_model_cfg,
    options_cfg=options_cfg,
    max_iters=5000)

checkpoint_config = dict(interval=1000)

fp16 = dict(loss_scale=256.0, velocity_accum_type='half', accum_type='half')
