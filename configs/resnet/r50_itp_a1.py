_base_ = [
    '../_base_/datasets/imagenet_bs256_itp_a12.py',
    '../_base_/schedules/imagenet_bs2048_itp.py',
    '../_base_/default_runtime.py'
]

# model settings
model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(3, ),
        norm_cfg=dict(type='BN', requires_grad=True),
        drop_path_rate=0.05,
        style='pytorch'),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='LinearClsHead',
        num_classes=1000,
        in_channels=2048,
        loss=dict(
            type='LabelSmoothLoss', label_smooth_val=0.1, mode='multi_label'),
        topk=(1, 5),
    ),
    train_cfg=dict(augments=[
        dict(type='BatchMixup', alpha=0.2, num_classes=1000, prob=0.5),
        dict(type='BatchCutMix', alpha=1.0, num_classes=1000, prob=0.5)
    ]))

optimizer = dict(lr=0.005, weight_decay=0.01)
fp16 = dict(loss_scale='dynamic')
runner = dict(max_epochs=600)
rep_aug = True
checkpoint_config = dict(interval=1, max_keep_ckpts=10)
