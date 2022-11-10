# Copyright (c) OpenMMLab. All rights reserved
import warnings

import numpy as np
from mmengine.hooks import Hook

from ...models.heads import ArcFaceClsHead


class SetFreqPowAdaptiveMarginHook(Hook):
    """Set margin based on frequency in ArcFaceClsHead Hook.

    Args:
    """

    def __init__(self, ground=0.05, range=0.45, power=-0.25) -> None:

        self.ground = ground
        self.range = range
        self.p = power

    def before_train(self, runner):
        """change the margins in ArcFaceClsHead.

        Args:
            runner (obj: `Runner`): Runner.
        """
        if not isinstance(runner.model.head, ArcFaceClsHead):
            warnings.warn('The head must be ``ArcFaceClsHead``.')
            return

        # generate margins base on the dataset.
        gt_labels = runner.train_dataloader.dataset.get_gt_labels()
        pow_freq = np.power(np.bincount(gt_labels), self.p)
        min_f, max_f = pow_freq.min(), pow_freq.max()
        margins = (pow_freq - min_f) / (max_f -
                                        min_f) * self.range + self.ground

        assert len(margins) == runner.model.head.num_classes
        runner.model.head.set_margin(margins)

        runner.model.head.with_adaptive_margin = True
