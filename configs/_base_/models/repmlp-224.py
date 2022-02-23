# model settings
model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='RepMLPNet',
        arch='B',
        img_size=224,
        out_indices=(3, ),
        reparam_conv_kernels=(1, 3),
        deploy=False),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='LinearClsHead',
        num_classes=1000,
        in_channels=768,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        topk=(1, 5),
    ))

if __name__ == '__main__':
    from mmcls.models import build_classifier
    import torch

    m = build_classifier(model).cuda()
    m.backbone.switch_to_deploy()

    x = torch.randn((1, 3, 224, 224)).cuda()
    y = m(x, return_loss=False, post_process=False)
    print(y.size())
