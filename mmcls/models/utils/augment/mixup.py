from abc import ABCMeta, abstractmethod

import numpy as np
import torch
import torch.nn.functional as F

from .builder import AUGMENT


class BaseMixupLayer(object, metaclass=ABCMeta):
    """Base class for MixupLayer.

    Args:
        alpha (float): Parameters for Beta distribution.
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
    """Mixup layer for batch mixup."""

    def __init__(self, *args, **kwargs):
        super(BatchMixupLayer, self).__init__(*args, **kwargs)

    def mixup(self, img, gt_label):
        one_hot_gt_label = F.one_hot(gt_label, num_classes=self.num_classes)
        lam = np.random.beta(self.alpha, self.alpha)
        batch_size = img.size(0)
        index = torch.randperm(batch_size)

        mixed_img = lam * img + (1 - lam) * img[index, :]
        mixed_gt_label = lam * one_hot_gt_label + (
            1 - lam) * one_hot_gt_label[index, :]

        return mixed_img, mixed_gt_label

    def __call__(self, img, gt_label):
        return self.mixup(img, gt_label)
