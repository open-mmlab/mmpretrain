# Copyright (c) OpenMMLab. All rights reserved.

import torch
from mmengine.hooks import Hook

from mmcls.registry import HOOKS


def pnorm(weights, p):
    normB = torch.norm(weights, 2, 1)
    ws = weights.clone()
    for i in range(weights.size(0)):
        ws[i] = ws[i] / torch.pow(normB[i], p)
    return ws


@HOOKS.register_module()
class TauNormHook(Hook):
    """rectify imbalance of decision boundaries by adjusting the classifier
    weight norms directly through the τ-normalization procedure.

    Args:
        tau (float):

    Example:
        To use this hook in config files.

        .. code:: python

            custom_hooks = [
                dict(
                    type='TauNormHook',
                    tau=0.7,
                )
            ]
    """
    priority = 'NORMAL'

    def __init__(self, tau: float):
        self.tau = tau

    def after_load_checkpoint(self, runner, checkpoint: dict) -> None:
        """All subclasses should override this method, if they need any
        operations after loading the checkpoint.

        Args:
            runner (Runner): The runner of the training, validation or testing
                process.
            checkpoint (dict): Model's checkpoint.
        """
        state_dict = checkpoint['state_dict']
        for layer_name in state_dict:
            if 'head' in layer_name and 'bias' not in layer_name:
                state_dict[layer_name] = pnorm(state_dict[layer_name],
                                               self.tau)
