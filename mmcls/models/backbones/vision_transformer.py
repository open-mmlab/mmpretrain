import torch
import torch.nn as nn
from mmcv.cnn import build_conv_layer, build_norm_layer
from mmcv.cnn.bricks.registry import (TRANSFORMER_LAYER,
                                      TRANSFORMER_LAYER_SEQUENCE)
from mmcv.cnn.bricks.transformer import (BaseTransformerLayer,
                                         TransformerLayerSequence,
                                         build_transformer_layer_sequence)
from mmcv.runner.base_module import BaseModule

from ..builder import BACKBONES
from ..utils import to_2tuple
from .base_backbone import BaseBackbone


@TRANSFORMER_LAYER.register_module()
class VitTransformerEncoderLayer(BaseTransformerLayer):
    """Implements encoder layer in Vit transformer.

    Args:
        ffn_dropout (float): Probability of an element to be zeroed
            in ffn. Default 0.0.
        act_cfg (dict): The activation config for FFNs.
    """

    def __init__(self, *args, **kwargs):
        super(VitTransformerEncoderLayer, self).__init__(*args, **kwargs)
        assert len(self.operation_order) == 4
        assert set(self.operation_order) == set(['self_attn', 'norm', 'ffn'])

    def init_weights(self):
        super(VitTransformerEncoderLayer, self).init_weights()
        for ffn in self.ffns:
            for m in ffn.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_normal_(m.weight)
                    nn.init.normal_(m.bias, std=1e-6)


@TRANSFORMER_LAYER_SEQUENCE.register_module()
class VitTransformerEncoder(TransformerLayerSequence):
    """TransformerEncoder of Vit.

    Args:
        coder_norm_cfg (dict): Config of last normalization layer. Default：
            `LN`. Only used when `self.pre_norm` is `True`
    """

    def __init__(
            self,
            *args,
            coder_norm_cfg=dict(type='LN'),
            **kwargs,
    ):
        super(VitTransformerEncoder, self).__init__(*args, **kwargs)
        if coder_norm_cfg is not None:
            self.coder_norm = build_norm_layer(
                coder_norm_cfg, self.embed_dims)[1] if self.pre_norm else None
        else:
            assert not self.pre_norm, f'Use prenorm in ' \
                                      f'{self.__class__.__name__},' \
                                      f'Please specify coder_norm_cfg'
            self.coder_norm = None

    def forward(self, *args, **kwargs):
        """Forward function for `TransformerCoder`.

        Returns:
            Tensor: forwarded results with shape [num_query, bs, embed_dims].
        """
        x = super(VitTransformerEncoder, self).forward(*args, **kwargs)
        if self.coder_norm is not None:
            x = self.coder_norm(x)
        return x


# Modified from pytorch-image-models
class PatchEmbed(BaseModule):
    """Image to Patch Embedding.

    Args:
        img_size (int | tuple): The size of input image.
        patch_size (int): The size of one patch
        in_channels (int): The num of input channels.
        embed_dim (int): The dimensions of embedding.
        norm_cfg (dict, optional): Config dict for normalization layer.
        conv_cfg (dict, optional): The config dict for conv layers.
            Default: None.
        init_cfg (`mmcv.ConfigDict`, optional): The Config for initialization.
            Default: None.
    """

    def __init__(self,
                 img_size=224,
                 patch_size=16,
                 in_channels=3,
                 embed_dim=768,
                 norm_cfg=None,
                 conv_cfg=None,
                 init_cfg=None):
        super(PatchEmbed, self).__init__(init_cfg)
        if isinstance(img_size, int):
            img_size = to_2tuple(img_size)
        elif isinstance(img_size, tuple):
            if len(img_size) == 1:
                img_size = to_2tuple(img_size[0])
            assert len(img_size) == 2, \
                f'The size of image should have length 1 or 2, ' \
                f'but got {len(img_size)}'

        self.img_size = img_size
        self.patch_size = to_2tuple(patch_size)

        patches_resolution = [
            img_size[0] // self.patch_size[0],
            img_size[1] // self.patch_size[1]
        ]
        num_patches = patches_resolution[0] * patches_resolution[1]
        assert num_patches * self.patch_size[0] * self.patch_size[1] == \
               self.img_size[0] * self.img_size[1], \
               'The image size H*W must be divisible by patch size'
        self.patches_resolution = patches_resolution
        self.num_patches = num_patches

        # Use conv layer to embed
        self.projection = build_conv_layer(
            conv_cfg,
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size)

        if norm_cfg is not None:
            self.norm = build_norm_layer(norm_cfg, self.embed_dims)[1]
        else:
            self.norm = None

    def forward(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't " \
            f'match model ({self.img_size[0]}*{self.img_size[1]}).'
        # The output size is (B, N, D), where N=H*W/P/P, D is embid_dim
        x = self.projection(x).flatten(2).transpose(1, 2)

        if self.norm is not None:
            x = self.norm(x)

        return x


# Modified from pytorch-image-models
class HybridEmbed(BaseModule):
    """CNN Feature Map Embedding.

    Extract feature map from CNN, flatten, project to embedding dim.
    """

    def __init__(self,
                 backbone,
                 img_size=224,
                 feature_size=None,
                 in_channels=3,
                 embed_dim=768,
                 conv_cfg=None,
                 init_cfg=None):
        super(HybridEmbed, self).__init__(init_cfg)
        assert isinstance(backbone, nn.Module)
        if isinstance(img_size, int):
            img_size = to_2tuple(img_size)
        elif isinstance(img_size, tuple):
            if len(img_size) == 1:
                img_size = to_2tuple(img_size[0])
            assert len(img_size) == 2, \
                f'The size of image should have length 1 or 2, ' \
                f'but got {len(img_size)}'

        self.img_size = img_size
        self.backbone = backbone
        if feature_size is None:
            with torch.no_grad():
                # FIXME this is hacky, but most reliable way of
                #  determining the exact dim of the output feature
                #  map for all networks, the feature metadata has
                #  reliable channel and stride info, but using
                #  stride to calc feature dim requires info about padding of
                #  each stage that isn't captured.
                training = backbone.training
                if training:
                    backbone.eval()
                o = self.backbone(
                    torch.zeros(1, in_channels, img_size[0], img_size[1]))
                if isinstance(o, (list, tuple)):
                    # last feature if backbone outputs list/tuple of features
                    o = o[-1]
                feature_size = o.shape[-2:]
                feature_dim = o.shape[1]
                backbone.train(training)
        else:
            feature_size = to_2tuple(feature_size)
            if hasattr(self.backbone, 'feature_info'):
                feature_dim = self.backbone.feature_info.channels()[-1]
            else:
                feature_dim = self.backbone.num_features
        self.num_patches = feature_size[0] * feature_size[1]

        # Use conv layer to embed
        self.projection = build_conv_layer(
            conv_cfg, feature_dim, embed_dim, kernel_size=1, stride=1)

    def forward(self, x):
        x = self.backbone(x)
        if isinstance(x, (list, tuple)):
            # last feature if backbone outputs list/tuple of features
            x = x[-1]
        x = self.projection(x).flatten(2).transpose(1, 2)
        return x


@BACKBONES.register_module()
class VisionTransformer(BaseBackbone):
    """ Vision Transformer
    A PyTorch impl of : `An Image is Worth 16x16 Words:
    Transformers for Image Recognition at Scale`  -
        https://arxiv.org/abs/2010.11929
    Args:
        embed_dim (int): Embedding dimension
        img_size (int | tuple): Input image size
        patch_size (int | tuple): The patch size
        in_channels (int): Number of input channels
        feedforward_channels (int): The hidden dimension for FFNs.
        drop_rate (float): Probability of an element to be zeroed.
            Default 0.0.
        hybrid_backbone (nn.Module, optional): CNN backbone to use in-place of
            PatchEmbed module. Default None.
        encoder (`mmcv.ConfigDict` | Dict): Config of TransformerEncoder
        init_cfg (dict, optional): Initialization config dict.
    """

    def __init__(self,
                 embed_dim=768,
                 img_size=224,
                 patch_size=16,
                 in_channels=3,
                 drop_rate=0.,
                 hybrid_backbone=None,
                 encoder=dict(
                     type='VitTransformerEncoder',
                     transformerlayers=None,
                     num_encoder_layers=12,
                     coder_norm_cfg=None,
                 ),
                 init_cfg=None):
        super(VisionTransformer, self).__init__(init_cfg)
        self.embed_dim = embed_dim

        if hybrid_backbone is not None:
            self.patch_embed = HybridEmbed(
                hybrid_backbone,
                img_size=img_size,
                in_channels=in_channels,
                embed_dim=embed_dim)
        else:
            self.patch_embed = PatchEmbed(
                img_size=img_size,
                patch_size=patch_size,
                in_channels=in_channels,
                embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, embed_dim))
        self.drop_after_pos = nn.Dropout(p=drop_rate)

        self.encoder = build_transformer_layer_sequence(encoder)

    def init_weights(self, pretrained=None):
        super(VisionTransformer, self).init_weights(pretrained)

        if pretrained is None:
            # Modified from ClassyVision
            nn.init.normal_(self.pos_embed, std=0.02)

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(
            B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.drop_after_pos(x)

        # Reason for permute:
        # as the shape of input x is [batch_size, num_query, embed_dim],
        # but the one from nn.MultiheadAttention is
        # [num_query, batch_size, embed_dim]
        x = x.permute(1, 0, 2)
        x = self.encoder(query=x, key=None, value=None)
        # Convert the shape back to [batch_size, num_query, embed_dim]
        x = x.permute(1, 0, 2)

        return x[:, 0]
