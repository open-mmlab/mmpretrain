from abc import ABCMeta, abstractmethod

import torch
import torch.nn.functional as F
from torch.distributions.beta import Beta


class BaseMixupLayer(object, metaclass=ABCMeta):
    """Base class for MixupLayer"""

    def __init__(self):
        super(BaseMixupLayer, self).__init__()

    @abstractmethod
    def mixup(self, imgs, gt_label):
        pass


class BatchMixupLayer(BaseMixupLayer):
    """Mixup layer for batch mixup.

    Args:
        alpha (float): Parameters for Beta distribution.
        num_classes (int): The number of classes.
    """

    def __init__(self, alpha, num_classes):
        super(BatchMixupLayer, self).__init__()

        assert isinstance(alpha, float)
        assert isinstance(num_classes, int)

        self.alpha = alpha
        self.num_classes = num_classes

        self.Beta = Beta(self.alpha, self.alpha)

    def mixup(self, img, gt_label):
        lam = self.Beta.sample()
        batch_size = img.size(0)
        index = torch.randperm(batch_size)
        one_hot_gt_label = F.one_hot(gt_label, num_classes=self.num_classes)

        mixed_img = lam * img + (1 - lam) * img[index, :]
        mixed_gt_label = lam * one_hot_gt_label + (
            1 - lam) * one_hot_gt_label[index, :]
        return mixed_img, mixed_gt_label

    def __call__(self, img, gt_label):
        return self.mixup(img, gt_label)
