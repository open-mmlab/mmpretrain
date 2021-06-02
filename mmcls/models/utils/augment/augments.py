import random

import numpy as np

from .builder import build_augment


class Augments:
    """Data augments.

    We implement some data augment methods, such as mixup, cutmix.
    Example:
        >>> augments_cfg = [
                dict(type='BatchCutMix', alpha=1., num_classes=10, prob=0.5),
                dict(type='BatchMixup', alpha=1., num_classes=10, prob=0.3),
                dict(type='Identity', num_classes=10, prob=0.2)
            ]
        >>> augments = Augments(augments_cfg)
        >>> imgs = torch.randn(16, 3, 32, 32)
        >>> label = torch.randint(0, 10, (16, ))
        >>> imgs, label = augments(imgs, label)

    To decide which augmentation within Augments block is used
    the following rule is applied.
    We pick augmentation based on the probabilities. In the example above,
    we decide if we should use BatchCutMix with probability 0.5,
    BatchMixup 0.3 and Identity otherwise 0.2.

    Args:
        augments_cfg (list[`mmcv.ConfigDict`] | obj:`mmcv.ConfigDict`):
            Config dict of augments.
    """

    def __init__(self, augments_cfg):
        super(Augments, self).__init__()

        if isinstance(augments_cfg, dict):
            augments_cfg = [augments_cfg]

        self.augments = [build_augment(cfg) for cfg in augments_cfg]
        self.augments_ps = [aug.prob for aug in self.augments]
        assert sum(self.augments_ps) == 1.0,\
            'The sum of augmentation probabilities should equal to 1,' \
            ' but got {:.2f}'.format(sum(self.augments_ps))

    def __call__(self, img, gt_label):
        if self.augments:
            random_state = np.random.RandomState(random.randint(0, 2**32 - 1))
            aug = random_state.choice(self.augments, p=self.augments_ps)
            return aug(img, gt_label)
        return img, gt_label
