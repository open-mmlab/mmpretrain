_base_ = [
    '../_base_/datasets/imagenet_bs64_pil_resize.py',
    '../_base_/schedules/imagenet_bs256_coslr.py',
    '../_base_/default_runtime.py'
]

model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='MobileOne',
        arch='s4',
        out_indices=(3, ),
    ),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='LinearClsHead',
        num_classes=1000,
        in_channels=2048,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        topk=(1, 5),
    ))

# dataset settings
data = dict(samples_per_gpu=128, workers_per_gpu=5)

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
