import math
import torch
import torch.nn as nn
import os
import hashlib
import urllib
from tqdm import tqdm
import warnings
from typing import List, Optional, Tuple, Dict

from mmpretrain.models.backbones.hivit import HiViT, PatchMerge, BlockWithRPE
from mmengine.model.weight_init import trunc_normal_
from .base import BaseSelfSupervisor
from mmpretrain.structures import DataSample
from ..utils import build_2d_sincos_position_embedding
from mmpretrain.models.utils.clip_generator_helper import build_clip_model

from mmpretrain.registry import MODELS


_MODELS = {
    "RN50": "https://openaipublic.azureedge.net/clip/models/afeb0e10f9e5a86da6080e35cf09123aca3b358a0c3e3b6c78a7b63bc04b6762/RN50.pt",
    "RN101": "https://openaipublic.azureedge.net/clip/models/8fa8567bab74a42d41c5915025a8e4538c3bdbe8804a470a72f30b0d94fab599/RN101.pt",
    "RN50x4": "https://openaipublic.azureedge.net/clip/models/7e526bd135e493cef0776de27d5f42653e6b4c8bf9e0f653bb11773263205fdd/RN50x4.pt",
    "RN50x16": "https://openaipublic.azureedge.net/clip/models/52378b407f34354e150460fe41077663dd5b39c54cd0bfd2b27167a4a06ec9aa/RN50x16.pt",
    "ViT-B/32": "https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt",
    "ViT-B/16": "https://openaipublic.azureedge.net/clip/models/5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/ViT-B-16.pt",
    "ViT-L/14": "https://openaipublic.azureedge.net/clip/models/b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836/ViT-L-14.pt",
    "ViT-L/14@336px": "https://openaipublic.azureedge.net/clip/models/3035c92b350959924f9f00213499208652fc7ea050643e8b385c2dac08641f02/ViT-L-14-336px.pt",
}


@MODELS.register_module()
class ClipTargeter(nn.Module):
    """Vector-Quantized Knowledge Distillation.

    The module only contains encoder and VectorQuantizer part
    Modified from https://github.com/microsoft/unilm/blob/master/beit2/modeling_vqkd.py

    Args:
        encoder_config (dict): The config of encoder.
        decoder_config (dict, optional): The config of decoder. Currently,
            VQKD only support to build encoder. Defaults to None.
        num_embed (int): Number of embedding vectors in the codebook. Defaults
            to 8192.
        embed_dims (int) : The dimension of embedding vectors in the codebook.
            Defaults to 32.
        decay (float): The decay parameter of EMA. Defaults to 0.99.
        beta (float): The mutiplier for VectorQuantizer loss. Defaults to 1.
        quantize_kmeans_init (bool): Whether to use k-means to initialize the
            VectorQuantizer. Defaults to True.
        init_cfg (dict or List[dict], optional): Initialization config dict.
            Defaults to None.
    """  # noqa: E501

    def __init__(
            self,
            model_name: str = 'ViT-B/16',
            jit: bool = False,
            checkpoint: str = None,
    ) -> None:
        super().__init__()
        if model_name in _MODELS:
            model_path = self._download(_MODELS[model_name], checkpoint or os.path.expanduser("~/.cache/clip"))
        elif os.path.isfile(model_name):
            model_path = model_name
        else:
            raise RuntimeError(f"Model {model_name} not found; available models = {list(_MODELS.keys())}")

        try:
            # loading JIT archive
            model = torch.jit.load(model_path, map_location="cpu").eval()
            state_dict = None
        except RuntimeError:
            # loading saved state dict
            if jit:
                warnings.warn(f"File {model_path} is not a JIT archive. Loading as a state dict instead")
                # jit = False
            state_dict = torch.load(model_path, map_location="cpu")

        # if not jit:
        self.clip_model = build_clip_model(state_dict or model.state_dict()).to('cuda')

    def _download(self, url: str, root: str):
        os.makedirs(root, exist_ok=True)
        filename = os.path.basename(url)

        expected_sha256 = url.split("/")[-2]
        download_target = os.path.join(root, filename)

        if os.path.exists(download_target) and not os.path.isfile(download_target):
            raise RuntimeError(f"{download_target} exists and is not a regular file")

        if os.path.isfile(download_target):
            if hashlib.sha256(open(download_target, "rb").read()).hexdigest() == expected_sha256:
                return download_target
            else:
                warnings.warn(f"{download_target} exists, but the SHA256 checksum does not match; re-downloading the file")

        with urllib.request.urlopen(url) as source, open(download_target, "wb") as output:
            with tqdm(total=int(source.info().get("Content-Length")), ncols=80, unit='iB', unit_scale=True,
                      unit_divisor=1024) as loop:
                while True:
                    buffer = source.read(8192)
                    if not buffer:
                        break

                    output.write(buffer)
                    loop.update(len(buffer))

        if hashlib.sha256(open(download_target, "rb").read()).hexdigest() != expected_sha256:
            raise RuntimeError(f"Model has been downloaded but the SHA256 checksum does not not match")

        return download_target

    def encode_image(self,
                     image: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode the image.

        Get the feature and attention mask from the last layer of the visual
        branch of CLIP.

        Args:
            image (torch.Tensor): The image tensor with shape NCHW.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: The feature and attention mask.
        """
        return self.clip_model.visual(image)[0]

    def forward(self,
                image: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode the image.

        Get the feature and attention mask from the last layer of the visual
        branch of CLIP.

        Args:
            image (torch.Tensor): The image tensor with shape NCHW.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: The feature and attention mask.
        """
        return self.encode_image(image)


@MODELS.register_module()
class MIMiTPN(HiViT):
    """Vision Transformer for MAE pre-training using HiViT.

        A PyTorch implement of: `An Image is Worth 16x16 Words: Transformers
        for Image Recognition at Scale <https://arxiv.org/abs/2010.11929>`_.
        This module implements the patch masking in MAE and initialize the
        position embedding with sine-cosine position embedding.

        Args:
            img_size (int | tuple): Input image size
            patch_size (int | tuple): The patch size
            drop_rate (float): Probability of an element to be zeroed.
                Defaults to 0.
            drop_path_rate (float): stochastic depth rate. Defaults to 0.
            mask_ratio (bool): The ratio of total number of patches to be masked.
                Defaults to 0.75.
        """

    def __init__(self,
                 img_size: int = 224,
                 patch_size: int = 16,
                 inner_patches: int = 4,
                 embed_dim: int = 512,
                 depths: list = [2, 2, 24],
                 num_heads: int = 8,
                 stem_mlp_ratio: int = 3.,
                 mlp_ratio: int = 4.,
                 qkv_bias: bool = True,
                 qk_scale: Optional[bool] = None,
                 drop_rate: float = 0.0,
                 attn_drop_rate: float = 0.0,
                 drop_path_rate: float = 0.0,
                 norm_cfg: dict = dict(type='LN', eps=1e-6),
                 ape: bool = True,
                 rpe: bool = False,
                 layer_scale_init_value: float = 0.0,
                 mask_ratio: float = 0.75,
                 reconstruction_type: str = 'pixel',  # iTPN supports pixel or clip as supervision
                 ):
        super().__init__(
            img_size=img_size,
            patch_size=patch_size,
            inner_patches=inner_patches,
            embed_dim=embed_dim,
            depths=depths,
            num_heads=num_heads,
            stem_mlp_ratio=stem_mlp_ratio,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            norm_cfg=norm_cfg,
            ape=ape,
            rpe=rpe,
            layer_scale_init_value=layer_scale_init_value
        )

        self.pos_embed.requires_grad = False
        self.mask_ratio = mask_ratio
        self.reconstruction_type = reconstruction_type
        self.num_patches = self.patch_embed.num_patches

        if reconstruction_type == 'clip':
            self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

    def init_weights(self) -> None:
        """Initialize position embedding, patch embedding and cls token."""
        super().apply(self._init_weights)

        if self.reconstruction_type == 'clip':
            trunc_normal_(self.mask_token, std=0.02)
            self.rescale_init_weight()
        else:
            pos_embed = build_2d_sincos_position_embedding(
                int(self.num_patches ** .5),
                self.pos_embed.shape[-1],
                cls_token=False)
            self.pos_embed.data.copy_(pos_embed.float())

            w = self.patch_embed.proj.weight.data
            torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

    def rescale_init_weight(self) -> None:
        """Rescale the initialized weights."""

        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))

        for layer_id, layer in enumerate(self.blocks):
            if isinstance(layer, BlockWithRPE):
                if layer.attn is not None:
                    rescale(layer.attn.proj.weight.data, layer_id + 1)
                rescale(layer.mlp.fc2.weight.data, layer_id + 1)

    def masking_id(self, batch_size, mask_ratio):
        N, L = batch_size, self.pos_embed.size(1)
        len_keep = int(L * (1 - mask_ratio))

        noise = torch.rand(N, L, device=self.pos_embed.device)  # noise in [0, 1]

        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=self.pos_embed.device)
        mask[:, :ids_keep.size(1)] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return ids_keep, ids_restore, mask

    def forward_pixel(
            self,
            x: torch.Tensor,
            mask: Optional[bool] = True
    ) -> Tuple[Tuple, torch.Tensor, torch.Tensor]:
        """Generate features for masked images.

        The function supports two kind of forward behaviors. If the ``mask`` is
        ``True``, the function will generate mask to masking some patches
        randomly and get the hidden features for visible patches, which means
        the function will be executed as masked imagemodeling pre-training;
        if the ``mask`` is ``None`` or ``False``, the forward function will
        call ``super().forward()``, which extract features from images without
        mask.


        Args:
            x (torch.Tensor): Input images, which is of shape B x C x H x W.
            mask (bool, optional): To indicate whether the forward function
                generating ``mask`` or not.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Hidden features,
            mask and the ids to restore original image.

            - ``x`` (torch.Tensor): hidden features, which is of shape
              B x (L * mask_ratio) x C.
            - ``mask`` (torch.Tensor): mask used to mask image.
            - ``ids_restore`` (torch.Tensor): ids to restore original image.
        """
        if mask is None or False:
            return super().forward(x)

        else:
            B, C, H, W = x.shape
            ids_keep, ids_restore, mask = self.masking_id(B, self.mask_ratio)

            x = self.patch_embed(x)

            x = torch.gather(
                x, dim=1, index=ids_keep[:, :, None, None, None].expand(-1, -1, *x.shape[2:])
            )

            outs = []
            for blk in self.blocks[:-self.num_main_blocks]:
                if isinstance(blk, PatchMerge):
                    outs.append(x)
                x = blk(x)

            x = x[..., 0, 0, :]
            if self.ape:
                pos_embed = self.interpolate_pos_encoding(x, H, W)
                pos_embed = torch.gather(
                    pos_embed.expand(B, -1, -1),
                    dim=1,
                    index=ids_keep[:, :, None].expand(-1, -1, pos_embed.shape[2]),
                )
                x = x + pos_embed
            x = self.pos_drop(x)

            for blk in self.blocks[-self.num_main_blocks:]:
                x = blk(x)

            outs.append(x)

            return (tuple(outs), mask, ids_restore)

    def forward_clip(
            self,
            x: torch.Tensor,
            mask: Optional[bool] = True
    ) -> Tuple:
        """Generate features for masked images.

        The function supports two kind of forward behaviors. If the ``mask`` is
        ``True``, the function will generate mask to masking some patches
        randomly and get the hidden features for visible patches, which means
        the function will be executed as masked imagemodeling pre-training;
        if the ``mask`` is ``None`` or ``False``, the forward function will
        call ``super().forward()``, which extract features from images without
        mask.


        Args:
            x (torch.Tensor): Input images, which is of shape B x C x H x W.
            mask (bool, optional): To indicate whether the forward function
                generating ``mask`` or not.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Hidden features,
            mask and the ids to restore original image.

            - ``x`` (torch.Tensor): hidden features, which is of shape
              B x (L * mask_ratio) x C.
            - ``mask`` (torch.Tensor): mask used to mask image.
            - ``ids_restore`` (torch.Tensor): ids to restore original image.
        """
        if mask is None or False:
            return super().forward(x)

        else:
            B, C, H, W = x.shape
            x = self.patch_embed(x)

            outs = []
            for blk in self.blocks[:-self.num_main_blocks]:
                if isinstance(blk, PatchMerge):
                    outs.append(x)
                x = blk(x)

            x = x[..., 0, 0, :]
            B, L, _ = x.shape
            mask_token = self.mask_token.expand(B, L, -1)
            w = mask.flatten(1).unsqueeze(-1).type_as(mask_token)
            x = x * (1. - w) + mask_token * w

            if self.ape:
                pos_embed = self.interpolate_pos_encoding(x, H, W)
                x = x + pos_embed
            x = self.pos_drop(x)

            rpe_index = True if self.rpe else None

            for blk in self.blocks[-self.num_main_blocks:]:
                x = blk(x, rpe_index)

            outs.append(x)

            return tuple(outs)

    def forward(
            self,
            x: torch.Tensor,
            mask: Optional[bool] = True
    ) -> Tuple:
        """Generate features for masked images.

        The function supports two kind of forward behaviors. If the ``mask`` is
        ``True``, the function will generate mask to masking some patches
        randomly and get the hidden features for visible patches, which means
        the function will be executed as masked imagemodeling pre-training;
        if the ``mask`` is ``None`` or ``False``, the forward function will
        call ``super().forward()``, which extract features from images without
        mask.


        Args:
            x (torch.Tensor): Input images, which is of shape B x C x H x W.
            mask (bool, optional): To indicate whether the forward function
                generating ``mask`` or not.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Hidden features,
            mask and the ids to restore original image.

            - ``x`` (torch.Tensor): hidden features, which is of shape
              B x (L * mask_ratio) x C.
            - ``mask`` (torch.Tensor): mask used to mask image.
            - ``ids_restore`` (torch.Tensor): ids to restore original image.
        """

        if self.reconstruction_type == 'pixel':
            return self.forward_pixel(x, mask)
        return self.forward_clip(x, mask)


@MODELS.register_module()
class iTPN(BaseSelfSupervisor):
    """iTPN.

    Implementation of `iTPN: Integrally Pre-Trained Transformer Pyramid Networks
    <https://arxiv.org/abs/2211.12735>`_.
    """

    def extract_feat(self, inputs: torch.Tensor):
        return self.backbone(inputs, mask=None)

    def loss(self, inputs: torch.Tensor, data_samples: List[DataSample],
             **kwargs) -> Dict[str, torch.Tensor]:
        """The forward function in training.

        Args:
            inputs (torch.Tensor): The input images.
            data_samples (List[DataSample]): All elements required
                during the forward function.

        Returns:
            Dict[str, torch.Tensor]: A dictionary of loss components.
        """

        if self.backbone.reconstruction_type == 'pixel':
            latent, mask, ids_restore = self.backbone(inputs)
            pred = self.neck(latent, ids_restore)

            loss = self.head.loss(pred, inputs, mask)
        else:
            mask = torch.stack([data_sample.mask for data_sample in data_samples])

            img_latent = self.backbone(inputs[0], mask)

            # inputs[1] is the target image
            with torch.no_grad():
                target = self.target_generator(inputs[1])
                target = target.detach()

            # iTPN contains a neck module
            feats = self.neck(img_latent)
            loss = self.head.loss(feats, target[:, 1:, :], mask)

        if isinstance(loss, torch.Tensor):
            losses = dict(loss=loss)
            return losses
        elif isinstance(loss, Tuple):
            # the loss_1 and loss_2 are general reconstruction loss (patch
            # feature vectors from last layer of backbone) and early state
            # reconstruction loss (patch feature vectors from intermediate
            # layer of backbone)
            loss_1, loss_2 = loss[0], loss[1]
            losses = dict()
            # the key with prefix 'loss', like loss_1 and loss_2, will be used
            # as the final criterion
            losses['loss_1'] = loss_1
            losses['loss_2'] = loss_2
            return losses
