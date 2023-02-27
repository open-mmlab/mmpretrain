# Copyright (c) OpenMMLab. All rights reserved.
from typing import Any, List, Tuple

import torch
from mmengine.dist import all_gather, get_rank


class GatherLayer(torch.autograd.Function):
    """Gather tensors from all process, supporting backward propagation."""

    @staticmethod
    def forward(ctx: Any, input: torch.Tensor) -> Tuple[List]:
        ctx.save_for_backward(input)
        output = all_gather(input)
        return tuple(output)

    @staticmethod
    def backward(ctx: Any, *grads: torch.Tensor) -> torch.Tensor:
        input, = ctx.saved_tensors
        grad_out = torch.zeros_like(input)
        grad_out[:] = grads[get_rank()]
        return grad_out
