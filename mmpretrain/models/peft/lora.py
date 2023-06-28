# Copyright (c) OpenMMLab. All rights reserved.
import math
import re
from typing import List, Optional

import torch
from mmengine.model import BaseModule
from torch import nn

from mmpretrain.registry import MODELS


class LoRALinear(nn.Module):
    """
    TODO
    """
    def __init__(self,
                 original_layer: nn.Linear,
                 alpha: int = 1,
                 rank: int = 0,
                 drop_rate: float = 0.):
        super(LoRALinear, self).__init__()
        in_features = original_layer.in_features
        out_features = original_layer.out_features

        self.lora_dropout = nn.Dropout(drop_rate)
        self.lora_down = nn.Linear(in_features, rank, bias=False)
        self.lora_up = nn.Linear(rank, out_features, bias=False)
        self.scaling = alpha / rank

        nn.init.kaiming_uniform_(self.lora_down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_down.weight)

    def forward(self, x: torch.Tensor):
        out = self.original_layer(x)

        lora_x = self.lora_dropout(x)
        lora_out = self.lora_up(self.lora_down(lora_x)) / self.scaling

        return out + lora_out


@MODELS.register_module()
class LoRAModel(BaseModule):
    """
    TODO
    """
    def __init__(self,
                 module: dict,
                 alpha: int = 1,
                 rank: int = 0,
                 drop_rate: float = 0.,
                 targets: Optional[List[dict]] = None):

        super().__init__(init_cfg=module['init_cfg'])

        module = MODELS.build(module)

        self.module: nn.Module = module
        self.alpha = alpha
        self.rank = rank
        self.drop_rate = drop_rate
        self.targets = targets

        self.apply_lora()

    def apply_lora(self):
        module_names = [k for k, _ in self.module.named_modules()]
        for module_name in module_names:
            for target in self.targets:
                target_name = target['type']
                target_alpha = target.get('alpha', self.alpha)
                target_rank = target.get('rank', self.rank)
                target_drop_rate = target.get('drop_rate', self.drop_rate)

                if re.fullmatch(target_name, module_name) or \
                        module_name.endswith(target_name):
                    child_module = self.module.get_submodule(module_name)
                    if isinstance(child_module, nn.Linear):
                        self._replace_module(module_name,
                                             child_module,
                                             target_alpha,
                                             target_rank,
                                             target_drop_rate)
                    else:
                        raise NotImplementedError

    def _replace_module(self,
                        module_name: str,
                        child_module: nn.Module,
                        alpha: int,
                        rank: int,
                        drop_rate: float):
        parent_module_name = ".".join(module_name.split(".")[:-1])
        parent_module = self.module.get_submodule(parent_module_name)

        target_name = module_name.split(".")[-1]
        target_module = LoRALinear(child_module,
                                   alpha,
                                   rank,
                                   drop_rate)
        setattr(parent_module, target_name, target_module)

    def _set_trainable(self):
        raise NotImplementedError

    def state_dict(self):
        raise NotImplementedError
