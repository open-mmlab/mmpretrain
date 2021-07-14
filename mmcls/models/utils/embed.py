import torch
import torch.nn as nn
from mmcv.cnn import build_conv_layer, build_norm_layer
from mmcv.runner.base_module import BaseModule

from .helpers import to_2tuple


class PatchEmbed(BaseModule):
    """Image to Patch Embedding.

    We use a conv layer to implement PatchEmbed.

    Args:
        img_size (int | tuple): The size of input image. Default: 224
        in_channels (int): The num of input channels. Default: 3
        embed_dims (int): The dimensions of embedding. Default: 768
        norm_cfg (dict, optional): Config dict for normalization layer.
            Default: None
        conv_cfg (dict, optional): The config dict for conv layers.
            Default: None
        init_cfg (`mmcv.ConfigDict`, optional): The Config for initialization.
            Default: None
    """

    def __init__(self,
                 img_size=224,
                 in_channels=3,
                 embed_dims=768,
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
        self.embed_dims = embed_dims

        # Use conv layer to embed
        conv_cfg = conv_cfg or dict()
        _conv_cfg = dict(
            type='Conv2d', kernel_size=16, stride=16, padding=0, dilation=1)
        _conv_cfg.update(conv_cfg)
        self.projection = build_conv_layer(_conv_cfg, in_channels, embed_dims)

        # Calculate how many patches a input image is splited to.
        h_out, w_out = [(self.img_size[i] + 2 * self.projection.padding[i] -
                         self.projection.dilation[i] *
                         (self.projection.kernel_size[i] - 1) - 1) //
                        self.projection.stride[i] + 1 for i in range(2)]

        self.patches_resolution = (h_out, w_out)
        self.num_patches = h_out * w_out

        if norm_cfg is not None:
            self.norm = build_norm_layer(norm_cfg, embed_dims)[1]
        else:
            self.norm = None

    def forward(self, x):
        B, C, H, W = x.shape
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

    Extract feature map from CNN, flatten,
    project to embedding dim.

    Args:
        backbone (nn.Module): CNN backbone
        img_size (int | tuple): The size of input image. Default: 224
        feature_size (int | tuple, optional): Size of feature map extracted by
            CNN backbone. Default: None
        in_channels (int): The num of input channels. Default: 3
        embed_dims (int): The dimensions of embedding. Default: 768
        conv_cfg (dict, optional): The config dict for conv layers.
            Default: None.
        init_cfg (`mmcv.ConfigDict`, optional): The Config for initialization.
            Default: None.
    """

    def __init__(self,
                 backbone,
                 img_size=224,
                 feature_size=None,
                 in_channels=3,
                 embed_dims=768,
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
        conv_cfg = conv_cfg or dict()
        _conv_cfg = dict(
            type='Conv2d', kernel_size=1, stride=1, padding=0, dilation=1)
        _conv_cfg.update(conv_cfg)
        self.projection = build_conv_layer(_conv_cfg, feature_dim, embed_dims)

    def forward(self, x):
        x = self.backbone(x)
        if isinstance(x, (list, tuple)):
            # last feature if backbone outputs list/tuple of features
            x = x[-1]
        x = self.projection(x).flatten(2).transpose(1, 2)
        return x


class PatchMerging(BaseModule):
    """Merge patch feature map.

    This layer use nn.Unfold to group feature map by kernel_size, and use norm
    and linear layer to embed grouped feature map.

    Args:
        input_resolution (tuple): The size of input patch resolution.
        in_channels (int): The num of input channels.
        expansion_ratio (Number): Expansion ratio of output channels. The num
            of output channels is equal to int(expansion_ratio * in_channels).
        kernel_size (int | tuple, optional): the kernel size in the unfold
            layer. Defaults to 2.
        stride (int | tuple, optional): the stride of the sliding blocks in the
            unfold layer. Defaults to be equal with kernel_size.
        padding (int | tuple, optional): zero padding width in the unfold
            layer. Defaults to 0.
        dilation (int | tuple, optional): dilation parameter in the unfold
            layer. Defaults to 1.
        bias (bool, optional): Whether to add bias in linear layer or not.
            Defaults to False.
        norm_cfg (dict, optional): Config dict for normalization layer.
            Defaults to dict(type='LN').
        init_cfg (dict, optional): The extra config for initialization.
            Defaults to None.
    """

    def __init__(self,
                 input_resolution,
                 in_channels,
                 expansion_ratio,
                 kernel_size=2,
                 stride=None,
                 padding=0,
                 dilation=1,
                 bias=False,
                 norm_cfg=dict(type='LN'),
                 init_cfg=None):
        super().__init__(init_cfg)
        H, W = input_resolution
        self.input_resolution = input_resolution
        self.in_channels = in_channels
        self.out_channels = int(expansion_ratio * in_channels)

        if stride is None:
            stride = kernel_size
        kernel_size = to_2tuple(kernel_size)
        stride = to_2tuple(stride)
        padding = to_2tuple(padding)
        dilation = to_2tuple(dilation)
        self.sampler = nn.Unfold(kernel_size, dilation, padding, stride)

        sample_dim = kernel_size[0] * kernel_size[1] * in_channels

        if norm_cfg is not None:
            self.norm = build_norm_layer(norm_cfg, sample_dim)[1]
        else:
            self.norm = None

        self.reduction = nn.Linear(sample_dim, self.out_channels, bias=bias)

        # See https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
        H_out = (H + 2 * padding[0] - dilation[0] *
                 (kernel_size[0] - 1) - 1) // stride[0] + 1
        W_out = (W + 2 * padding[1] - dilation[1] *
                 (kernel_size[1] - 1) - 1) // stride[1] + 1
        self.output_resolution = (H_out, W_out)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, 'input feature has wrong size'

        x = x.view(B, H, W, C).permute([0, 3, 1, 2])  # B, C, H, W

        # Use nn.Unfold to merge patch. About 25% faster than original method,
        # but need to modify pretrained model for compatibility
        x = self.sampler(x)  # B, 4*C, H/2*W/2
        x = x.transpose(1, 2)  # B, H/2*W/2, 4*C

        x = self.norm(x) if self.norm else x
        x = self.reduction(x)

        return x
