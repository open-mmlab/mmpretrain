import torch
from mmcv import Config
from torch.nn.modules import GroupNorm
from torch.nn.modules.batchnorm import _BatchNorm

from mmcls.models.backbones import VGG, VisionTransformer


def is_norm(modules):
    """Check if is one of the norms."""
    if isinstance(modules, (GroupNorm, _BatchNorm)):
        return True
    return False


def check_norm_state(modules, train_state):
    """Check if norm layer is in correct train state."""
    for mod in modules:
        if isinstance(mod, _BatchNorm):
            if mod.training != train_state:
                return False
    return True


def test_vit_backbone():

    model = dict(
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
                    type='FFN',
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
                nonlinearity='linear')
        ])
    cfg = Config(model)

    # Test ViT base model with input size of 224
    # and patch size of 16
    model = VisionTransformer(**cfg)
    model.init_weights()
    model.train()

    assert check_norm_state(model.modules(), True)

    imgs = torch.randn(3, 3, 224, 224)
    feat = model(imgs)
    assert feat.shape == torch.Size((3, 768))


def test_vit_hybrid_backbone():

    # Test VGG11+ViT-B/16 hybrid model
    backbone = VGG(11, norm_eval=True)
    backbone.init_weights()

    model = dict(
        embed_dim=768,
        img_size=224,
        patch_size=16,
        in_channels=3,
        drop_rate=0.1,
        hybrid_backbone=backbone,
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
                        dropout_layer=dict(type='DropOut', drop_prob=0.1))
                ],
                ffn_cfgs=dict(
                    type='FFN',
                    embed_dims=768,
                    feedforward_channels=3072,
                    num_fcs=2,
                    ffn_drop=0.1,
                    act_cfg=dict(type='GELU')),
                operation_order=('norm', 'self_attn', 'norm', 'ffn')),
            init_cfg=[
                dict(type='Xavier', layer='Linear', distribution='normal')
            ]),
        init_cfg=[
            dict(
                type='Kaiming',
                layer='Conv2d',
                mode='fan_in',
                nonlinearity='linear')
        ])
    cfg = Config(model)

    model = VisionTransformer(**cfg)
    model.init_weights()
    model.train()

    assert check_norm_state(model.modules(), True)

    imgs = torch.randn(1, 3, 224, 224)
    feat = model(imgs)
    assert feat.shape == torch.Size((1, 768))
