import random

import numpy as np

from .builder import build_augment


class Augments:
    """Data augments.

    We implement some data augment methods, such as mixup, cutmix.
    Example:
        >>> augments_cfg = [
                dict(type='BatchCutMix', alpha=1., num_classes=10, prob=1.),
                dict(type='BatchMixup', alpha=1., num_classes=10, prob=0.6),
                dict(type='Identity', num_classes=10, prob=0.4)
            ]
        >>> augments = Augments(augments_cfg)
        >>> imgs = torch.randn(16, 3, 32, 32)
        >>> label = torch.randint(0, 10, (16, ))
        >>> imgs, label = augments(imgs, label)

    To decide which augmentation within OneOf block is used
    the following rule is applied.
    We normalize all probabilities within augments_cfg to one. After this
    we pick augmentation based on the normalized probabilities. In the example
    above BatchCutMix has probability 1.0, BatchMixup probability 0.6 and
    Identity probability 0.4. After normalization, they become 0.5, 0.3
    and 0.2. Which means that we decide if we should use BatchCutMix with
    probability 0.5, BatchMixup 0.3 and Identity otherwise 0.2.

    Args:
        augments_cfg (list[`mmcv.ConfigDict`] | obj:`mmcv.ConfigDict`):
            Config dict of augments.
    """

    def __init__(self, augments_cfg):
        super(Augments, self).__init__()

        if isinstance(augments_cfg, dict):
            augments_cfg = [augments_cfg]

        self.augments = [build_augment(cfg) for cfg in augments_cfg]
        augments_ps = [aug.prob for aug in self.augments]
        s = sum(augments_ps)
        self.augments_ps = [a / s for a in augments_ps]

    def __call__(self, img, gt_label):
        if self.augments:
            random_state = np.random.RandomState(random.randint(0, 2**32 - 1))
            aug = random_state.choice(self.augments, p=self.augments_ps)
            return aug(img, gt_label)
        return img, gt_label
