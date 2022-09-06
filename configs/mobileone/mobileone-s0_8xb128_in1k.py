_base_ = [
    '../_base_/datasets/imagenet_bs64_pil_resize.py',
    '../_base_/schedules/imagenet_bs256_coslr.py',
    '../_base_/default_runtime.py'
]

# dataset settings
dataset_type = 'ImageNet'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', size=(256, -1), backend='pillow'),
    dict(type='CenterCrop', crop_size=224),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='Collect', keys=['img'])
]
data = dict(
    samples_per_gpu=128,
    workers_per_gpu=5,
    val=dict(pipeline=test_pipeline),
    test=dict(pipeline=test_pipeline))

model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='MobileOne',
        arch='s0',
        out_indices=(3, ),
    ),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='LinearClsHead',
        num_classes=1000,
        in_channels=1024,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        topk=(1, 5),
    ))

if __name__ == '__main__':
    import torch

    from mmcls.models import build_classifier
    x = torch.randn((1, 3, 224, 224))
    import numpy as np

    classifier = build_classifier(model)
    classifier.eval()
    for p, _ in classifier.named_parameters():
        print(p)
    y = classifier(x, return_loss=False)
    print(type(y))
    classifier.backbone.switch_to_deploy()
    print(classifier)
    y_ = classifier(x, return_loss=False)
    print(type(y), type(y[0]))
    assert np.allclose(y[0], y_[0]), (y[0][:20], y_[0][:20])
