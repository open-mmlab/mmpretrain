import torch
from mmcv import Config

from mmcls.models.classifiers import ImageClassifier


def test_image_classifier():

    # Test mixup in ImageClassifier
    model_cfg = dict(
        backbone=dict(
            type='ResNet_CIFAR',
            depth=50,
            num_stages=4,
            out_indices=(3, ),
            style='pytorch'),
        neck=dict(type='GlobalAveragePooling'),
        head=dict(
            type='MultiLabelLinearClsHead',
            num_classes=10,
            in_channels=2048,
            loss=dict(type='CrossEntropyLoss', loss_weight=1.0,
                      use_soft=True)),
        train_cfg=dict(mixup=dict(alpha=1.0, num_classes=10)))
    img_classifier = ImageClassifier(**model_cfg)
    img_classifier.init_weights()
    imgs = torch.randn(16, 3, 32, 32)
    label = torch.randint(0, 10, (16, ))

    losses = img_classifier.forward_train(imgs, label)
    assert losses['loss'].item() > 0


def test_image_classifier_with_cutmix():

    # Test cutmix in ImageClassifier
    model_cfg = dict(
        backbone=dict(
            type='ResNet_CIFAR',
            depth=50,
            num_stages=4,
            out_indices=(3, ),
            style='pytorch'),
        neck=dict(type='GlobalAveragePooling'),
        head=dict(
            type='MultiLabelLinearClsHead',
            num_classes=10,
            in_channels=2048,
            loss=dict(type='CrossEntropyLoss', loss_weight=1.0,
                      use_soft=True)),
        train_cfg=dict(
            cutmix=dict(alpha=1.0, num_classes=10, cutmix_prob=1.0)))
    img_classifier = ImageClassifier(**model_cfg)
    img_classifier.init_weights()
    imgs = torch.randn(16, 3, 32, 32)
    label = torch.randint(0, 10, (16, ))

    losses = img_classifier.forward_train(imgs, label)
    assert losses['loss'].item() > 0


def test_image_classifier_with_label_smooth_loss():

    # Test mixup in ImageClassifier
    model_cfg = dict(
        backbone=dict(
            type='ResNet_CIFAR',
            depth=50,
            num_stages=4,
            out_indices=(3, ),
            style='pytorch'),
        neck=dict(type='GlobalAveragePooling'),
        head=dict(
            type='MultiLabelLinearClsHead',
            num_classes=10,
            in_channels=2048,
            loss=dict(type='LabelSmoothLoss', label_smooth_val=0.1)),
        train_cfg=dict(mixup=dict(alpha=1.0, num_classes=10)))
    img_classifier = ImageClassifier(**model_cfg)
    img_classifier.init_weights()
    imgs = torch.randn(16, 3, 32, 32)
    label = torch.randint(0, 10, (16, ))

    losses = img_classifier.forward_train(imgs, label)
    assert losses['loss'].item() > 0


def test_image_classifier_vit():

    model_cfg = dict(
        backbone=dict(
            type='VisionTransformer',
            embed_dim=768,
            img_size=224,
            patch_size=16,
            in_channels=3,
            drop_rate=0.1,
            hybrid_backbone=None,
            encoder=dict(
                type='VitTransformerEncoder',
                num_layers=12,
                transformerlayers=dict(
                    type='VitTransformerEncoderLayer',
                    attn_cfgs=[
                        dict(
                            type='MultiheadAttention',
                            embed_dims=768,
                            num_heads=12,
                            attn_drop=0.,
                            proj_drop=0.1,
                            batch_first=True)
                    ],
                    ffn_cfgs=dict(
                        embed_dims=768,
                        feedforward_channels=3072,
                        num_fcs=2,
                        ffn_drop=0.1,
                        act_cfg=dict(type='GELU')),
                    operation_order=('norm', 'self_attn', 'norm', 'ffn'),
                    batch_first=True),
                init_cfg=[
                    dict(type='Xavier', layer='Linear', distribution='normal')
                ]),
            init_cfg=[
                dict(
                    type='Kaiming',
                    layer='Conv2d',
                    mode='fan_in',
                    nonlinearity='linear'),
                dict(
                    type='Pretrained',
                    checkpoint='../checkpoints/vit/vit_base_patch16_224.pth',
                    prefix='backbone.')
            ]),
        neck=None,
        head=dict(
            type='VisionTransformerClsHead',
            num_classes=1000,
            in_channels=768,
            hidden_dim=3072,
            loss=dict(type='LabelSmoothLoss', label_smooth_val=0.1),
            topk=(1, 5),
        ),
        train_cfg=dict(mixup=dict(alpha=0.2, num_classes=1000)))

    model_cfg = Config(model_cfg)
    img_classifier = ImageClassifier(**model_cfg)
    img_classifier.init_weights()

    # test initializing weights of a sub-module
    # with the specific part of a pretrained model by using 'prefix'
    checkpoint = torch.load(
        '../checkpoints/vit/vit_base_patch16_224.pth', map_location='cpu')
    assert (checkpoint['state_dict']['backbone.cls_token'] ==
            img_classifier.state_dict()['backbone.cls_token']).all()

    imgs = torch.randn(1, 3, 224, 224)
    label = torch.randint(0, 1000, (1, ))

    losses = img_classifier.forward_train(imgs, label)
    assert losses['loss'].item() > 0
