# Copyright (c) OpenMMLab. All rights reserved.
from abc import ABCMeta, abstractmethod

import numpy as np
import torch

from .builder import AUGMENT
from .utils import one_hot_encoding


class BaseMixupLayer(object, metaclass=ABCMeta):
    """Base class for MixupLayer.

    Args:
        alpha (float): Parameters for Beta distribution to generate the
            mixing ratio. It should be a positive number.
        num_classes (int): The number of classes.
        prob (float): MixUp probability. It should be in range [0, 1].
            Default to 1.0
    """

    def __init__(self, alpha, num_classes, prob=1.0):
        super(BaseMixupLayer, self).__init__()

        assert isinstance(alpha, float) and alpha > 0
        assert isinstance(num_classes, int)
        assert isinstance(prob, float) and 0.0 <= prob <= 1.0

        self.alpha = alpha
        self.num_classes = num_classes
        self.prob = prob

    @abstractmethod
    def mixup(self, imgs, gt_label):
        pass


@AUGMENT.register_module(name='BatchMixup')
class BatchMixupLayer(BaseMixupLayer):
    r"""Mixup layer for a batch of data.

    Mixup is a method to reduces the memorization of corrupt labels and
    increases the robustness to adversarial examples. It's
    proposed in `mixup: Beyond Empirical Risk Minimization
    <https://arxiv.org/abs/1710.09412>`

    This method simply linearly mix pairs of data and their labels.

    Args:
        alpha (float): Parameters for Beta distribution to generate the
            mixing ratio. It should be a positive number. More details
            are in the note.
        num_classes (int): The number of classes.
        prob (float): The probability to execute mixup. It should be in
            range [0, 1]. Default sto 1.0.

    Note:
        The :math:`\alpha` (``alpha``) determines a random distribution
        :math:`Beta(\alpha, \alpha)`. For each batch of data, we sample
        a mixing ratio (marked as :math:`\lambda`, ``lam``) from the random
        distribution.
    """

    def __init__(self, *args, **kwargs):
        super(BatchMixupLayer, self).__init__(*args, **kwargs)

    def mixup(self, img, gt_label):
        one_hot_gt_label = one_hot_encoding(gt_label, self.num_classes)
        lam = np.random.beta(self.alpha, self.alpha)
        batch_size = img.size(0)
        index = torch.randperm(batch_size)

        mixed_img = lam * img + (1 - lam) * img[index, :]
        mixed_gt_label = lam * one_hot_gt_label + (
            1 - lam) * one_hot_gt_label[index, :]

        return mixed_img, mixed_gt_label

    def __call__(self, img, gt_label):
        return self.mixup(img, gt_label)
