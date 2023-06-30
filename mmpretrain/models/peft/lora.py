# Copyright (c) OpenMMLab. All rights reserved.
import math
import re
from typing import List

import torch
from mmengine.logging import print_log
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

        self.original_layer = original_layer

    def init_weights(self):
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
                 targets: List[dict] = list()):

        super().__init__()

        module = MODELS.build(module)

        self.module: nn.Module = module
        self.alpha = alpha
        self.rank = rank
        self.drop_rate = drop_rate

        assert len(targets) != 0, \
            "The length of target layers should not be 0."

        self.targets = targets

        self.apply_lora()
        self._freeze_module()
        self._register_hooks()

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
                    current_module = self.module.get_submodule(module_name)
                    if isinstance(current_module, nn.Linear):
                        print_log(f'Set LoRA for {module_name} '
                                  f'with alpha: {target_alpha}, '
                                  f'rank: {target_rank}, '
                                  f'drop rate: {target_drop_rate}',
                                  logger='current')

                        self._replace_module(module_name, current_module,
                                             target_alpha, target_rank,
                                             target_drop_rate)

    def _replace_module(self, module_name: str, current_module: nn.Module,
                        alpha: int, rank: int, drop_rate: float):
        parent_module_name = ".".join(module_name.split(".")[:-1])
        parent_module = self.module.get_submodule(parent_module_name)

        target_name = module_name.split(".")[-1]
        target_module = LoRALinear(current_module, alpha, rank, drop_rate)
        setattr(parent_module, target_name, target_module)

    def _freeze_module(self):
        for name, param in self.named_parameters():
            if 'lora_' not in name:
                param.requires_grad = False

    def _register_hooks(self):

        def _state_dict_hook(module, state_dict, prefix, local_metadata):
            keys = [k for k, _ in state_dict.items()]
            for key in keys:
                if 'lora_' not in key:
                    state_dict.pop(key)

        self._register_state_dict_hook(_state_dict_hook)

        def _load_state_dict_post_hook(module, incompatible_keys):
            missing_keys = incompatible_keys.missing_keys.copy()
            for key in missing_keys:
                if 'lora_' not in key:
                    incompatible_keys.missing_keys.remove(key)

            unexpected_keys = incompatible_keys.unexpected_keys.copy()
            for key in unexpected_keys:
                if 'lora_' not in key:
                    incompatible_keys.unexpected_keys.remove(key)

        self.register_load_state_dict_post_hook(_load_state_dict_post_hook)
