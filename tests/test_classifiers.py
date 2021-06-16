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
            num_layers=12,
            embed_dim=768,
            num_heads=12,
            img_size=224,
            patch_size=16,
            in_channels=3,
            feedforward_channels=3072,
            drop_rate=0.1,
            attn_drop_rate=0.),
        neck=None,
        head=dict(
            type='VisionTransformerClsHead',
            num_classes=1000,
            in_channels=768,
            hidden_dim=3072,
            loss=dict(type='CrossEntropyLoss', loss_weight=1.0, use_soft=True),
            topk=(1, 5),
        ),
        train_cfg=dict(mixup=dict(alpha=0.2, num_classes=1000)))
    img_classifier = ImageClassifier(**model_cfg)
    img_classifier.init_weights()
    imgs = torch.randn(3, 3, 224, 224)
    label = torch.randint(0, 1000, (3, ))

    losses = img_classifier.forward_train(imgs, label)
    assert losses['loss'].item() > 0


def test_image_classifier_t2t_vit():
    model_cfg = dict(
        backbone=dict(
            type='T2T_ViT',
            t2t_module=dict(
                img_size=224,
                tokens_type='transformer',
                in_chans=3,
                embed_dim=384,
                token_dim=64),
            encoder=dict(
                type='T2TTransformerEncoder',
                num_layers=14,
                transformerlayers=dict(
                    type='T2TTransformerEncoderLayer',
                    attn_cfgs=dict(
                        type='T2TBlockAttention',
                        embed_dims=384,
                        num_heads=6,
                        attn_drop=0.,
                        proj_drop=0.,
                        dropout_layer=dict(type='DropPath')),
                    ffn_cfgs=dict(
                        type='FFN',
                        embed_dims=384,
                        feedforward_channels=3 * 384,
                        num_fcs=2,
                        act_cfg=dict(type='GELU'),
                        dropout_layer=dict(type='DropPath')),
                    operation_order=('norm', 'self_attn', 'norm', 'ffn'),
                    batch_first=True),
                drop_path_rate=0.1),
            init_cfg=[
                dict(type='TruncNormal', layer='Linear', std=.02),
                dict(type='Constant', layer='LayerNorm', val=1., bias=0.)
            ]),
        neck=None,
        head=dict(
            type='LinearClsHead',
            num_classes=1000,
            in_channels=384,
            loss=dict(type='LabelSmoothLoss', label_smooth_val=0.1),
            topk=(1, 5),
            init_cfg=dict(type='TruncNormal', layer='Linear', std=.02)),
        train_cfg=dict(
            cutmixup=dict(
                mixup_alpha=0.8,
                cutmix_alpha=1.0,
                prob=1.0,
                switch_prob=0.5,
                mode='batch',
                num_classes=1000)))

    model_cfg = Config(model_cfg)
    img_classifier = ImageClassifier(**model_cfg)
    img_classifier.init_weights()

    imgs = torch.randn(2, 3, 224, 224)
    label = torch.randint(0, 1000, (2, ))

    losses = img_classifier.forward_train(imgs, label)
    assert losses['loss'].item() > 0
