_base_ = [
    '../_base_/datasets/imagenet_bs128_poolformer_small_224.py',
    '../_base_/schedules/imagenet_bs1024_adamw_swin.py',
    '../_base_/default_runtime.py',
]

model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='EfficientFormer',
        arch='l7',
        drop_path_rate=0,
        init_cfg=[
            dict(
                type='TruncNormal',
                layer=['Conv2d', 'Linear'],
                std=.02,
                bias=0.),
            dict(type='Constant', layer=['GroupNorm'], val=1., bias=0.),
        ]),
    neck=dict(type='GlobalAveragePooling', dim=1),
    head=dict(
        type='EfficientFormerClsHead', in_channels=768, num_classes=1000))

if __name__ == '__main__':
    from mmcls.models import build_classifier
    m = build_classifier(model)
    print(m)
