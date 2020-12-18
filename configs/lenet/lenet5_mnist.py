# model settings
model = dict(
    type='ImageClassifier',
    backbone=dict(type='LeNet5', num_classes=10),
    neck=None,
    head=dict(
        type='ClsHead',
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
    ))
# dataset settings
dataset_type = 'MNIST'
img_norm_cfg = dict(mean=[33.46], std=[78.87], to_rgb=True)
train_pipeline = [
    dict(type='Resize', size=32),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='ToTensor', keys=['gt_label']),
    dict(type='Collect', keys=['img', 'gt_label']),
]
test_pipeline = [
    dict(type='Resize', size=32),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='Collect', keys=['img']),
]
data = dict(
    samples_per_gpu=128,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type, data_prefix='data/mnist', pipeline=train_pipeline),
    val=dict(
        type=dataset_type, data_prefix='data/mnist', pipeline=test_pipeline))
evaluation = dict(
    interval=5, metric='accuracy', metric_options={'topk': (1, )})
# optimizer
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(policy='step', step=[15])
# checkpoint saving
checkpoint_config = dict(interval=1)
# yapf:disable
log_config = dict(
    interval=150,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
# runtime settings
runner = dict(type='EpochBasedRunner', max_epochs=5)
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = './work_dirs/mnist/'
load_from = None
resume_from = None
workflow = [('train', 1)]
