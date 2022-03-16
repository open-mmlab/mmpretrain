_base_ = [
    '../_base_/datasets/imagenet_bs32_pil_resize.py',
    '../_base_/schedules/imagenet_bs256.py', '../_base_/default_runtime.py'
]

# model settings
model = dict(
    type='ImageClassifier',
    backbone=dict(type='CSPResNet', depth=50),
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

    model = build_classifier(model)
    model.eval()
    # print(model)
    inputs = torch.rand(1, 3, 224, 224)
    level_outputs = model.extract_feat(inputs, stage='backbone')
    for i, level_out in enumerate(level_outputs):
        print(i, tuple(level_out.shape))
