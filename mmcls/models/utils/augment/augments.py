import random

import numpy as np

from .builder import build_augment


class Augments:
    """Data augments.

    We implement some data augment methods, such as mixup, cutmix.
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
