import torch
from typing import List, Optional, Tuple, Dict, Union, Sequence

from mmpretrain.models import HiViT
from ..utils import build_2d_sincos_position_embedding
from .base import BaseSelfSupervisor
from mmpretrain.structures import DataSample
from mmpretrain.registry import MODELS


@MODELS.register_module()
class MIMHiViT(HiViT):
    """HiViT for MAE pre-training.

        A PyTorch implement of: ` HiViT: A Simple and More Efficient Design
        of Hierarchical Vision Transformer`.
        This module implements the patch masking in MAE and initialize the
        position embedding with sine-cosine position embedding.

        Args:
            arch (str | dict): Vision Transformer architecture
                Default: 'b'
            img_size (int | tuple): Input image size
            patch_size (int | tuple): The patch size
                Defaults to 4, to downsample 4x at the first stage
            inner_patches (int): The inner patches within a token
                Defaults to 4
            out_indices (Sequence | int): Output from which stages.
                Defaults to -1, means the last stage.
            drop_rate (float): Probability of an element to be zeroed.
                Defaults to 0.
            drop_path_rate (float): stochastic depth rate. Defaults to 0.
            norm_cfg (dict): Config dict for normalization layer.
                Defaults to ``dict(type='LN')``.
            ape (bool): the absolute position embedding
            rpe (bool): the relative position embedding
                Defaults to False
            layer_scale_init_value (float): the layer scale init value
            mask_ratio (bool): The ratio of total number of patches to be masked.
                Defaults to 0.75.
            init_cfg (Union[List[dict], dict], optional): Initialization config
                dict. Defaults to None.
        """

    def __init__(self,
                 arch: Union[str, dict] = 'b',
                 img_size: int = 224,
                 patch_size: int = 16,
                 inner_patches: int = 4,
                 out_indices: Union[list, int] = [23, ],
                 drop_rate: float = 0.0,
                 drop_path_rate: float = 0.0,
                 norm_cfg: dict = dict(type='LN', eps=1e-6),
                 ape: bool = True,
                 rpe: bool = False,
                 layer_scale_init_value: float = 0.0,
                 mask_ratio: float = 0.75,
                 init_cfg: Optional[Union[List[dict], dict]] = None) -> None:
        super().__init__(
            arch=arch,
            img_size=img_size,
            patch_size=patch_size,
            inner_patches=inner_patches,
            out_indices=out_indices,
            drop_rate=drop_rate,
            drop_path_rate=drop_path_rate,
            norm_cfg=norm_cfg,
            ape=ape,
            rpe=rpe,
            layer_scale_init_value=layer_scale_init_value,
            init_cfg=init_cfg
        )

        self.pos_embed.requires_grad = False
        self.mask_ratio = mask_ratio
        self.num_patches = self.patch_embed.num_patches

    def init_weights(self) -> None:
        """Initialize position embedding, patch embedding."""
        super().apply(self._init_weights)
        pos_embed = build_2d_sincos_position_embedding(
            int(self.num_patches ** .5),
            self.pos_embed.shape[-1],
            cls_token=False)
        self.pos_embed.data.copy_(pos_embed.float())

        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

    def masking_id(
            self,
            batch_size,
            mask_ratio
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """ Generate the mask for MAE Pre-training

        Args:
            batch_size: The batch size of input data
            mask_ratio: The mask ratio of total patches.
                Defaults to 0.75.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: the ids
            for the tokens retained, the ids to restore original image,
            and the mask
        """
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

    def forward(
            self,
            x: torch.Tensor,
            mask: Optional[bool] = True
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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

            for blk in self.blocks[:-self.num_main_blocks]:
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

            return (x, mask, ids_restore)
