# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional, Sequence

import torch
from mmengine.model import BaseModule

from mmcls.registry import MODELS


@MODELS.register_module()
class TransformerTokenMergeNeck(BaseModule):
    """


    Args:
        mode:
    """

    def __init__(self, mode: str = 'concat', init_cfg: Optional[dict] = None):
        super().__init__(init_cfg=init_cfg)
        assert mode in ['concat'], \
            f'Currently, TransformerTokenMergeNeck only supports "concat" ' \
            f'mode, but the input mode is "{mode}".'
        self.mode = mode

    def forward(self, inputs: Sequence[torch.Tensor]):
        outs = []
        for i in range(len(inputs)):
            patch_token, cls_token = inputs[i][0], inputs[i][1]
            if self.mode == 'concat':
                B, C, N1, N2 = patch_token.shape
                patch_token = patch_token.permute(0, 2, 3,
                                                  1).reshape(B, N1 * N2, C)
                patch_token_avg = torch.mean(patch_token, dim=1)
                merged_token = torch.cat((cls_token, patch_token_avg), dim=-1)
            outs.append(merged_token)
        return tuple(outs)
