from abc import ABCMeta, abstractmethod

import numpy as np
import torch
import torch.nn.functional as F


class BaseCutMixLayer(object, metaclass=ABCMeta):
    """Base class for CutMixLayer."""

    def __init__(self):
        super(BaseCutMixLayer, self).__init__()

    @abstractmethod
    def cutmix(self, imgs, gt_label):
        pass


class BatchCutMixLayer(BaseCutMixLayer):
    """CutMix layer for batch CutMix.

    Args:
        alpha (float): Parameters for Beta distribution. Positive(>0).
        num_classes (int): The number of classes.
        cutmix_prob (float): CutMix probability. It should be in range [0, 1]
    """

    def __init__(self, alpha, num_classes, cutmix_prob):
        super(BatchCutMixLayer, self).__init__()

        assert isinstance(alpha, float) and alpha > 0
        assert isinstance(num_classes, int)
        assert isinstance(cutmix_prob, float) and 0.0 <= cutmix_prob <= 1.0

        self.alpha = alpha
        self.num_classes = num_classes
        self.cutmix_prob = cutmix_prob

    def rand_bbox(self, size, lam):
        W = size[2]
        H = size[3]
        cut_rat = np.sqrt(1. - lam)
        cut_w = np.int(W * cut_rat)
        cut_h = np.int(H * cut_rat)

        # uniform
        cx = np.random.randint(W)
        cy = np.random.randint(H)

        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)

        return bbx1, bby1, bbx2, bby2

    def cutmix(self, img, gt_label):
        r = np.random.rand(1)
        if r < self.cutmix_prob:
            lam = np.random.beta(self.alpha, self.alpha)
            batch_size = img.size(0)
            index = torch.randperm(batch_size)
            one_hot_gt_label = F.one_hot(
                gt_label, num_classes=self.num_classes)
            bbx1, bby1, bbx2, bby2 = self.rand_bbox(img.size(), lam)
            img[:, :, bbx1:bbx2, bby1:bby2] = \
                img[index, :, bbx1:bbx2, bby1:bby2]
            # adjust lambda to exactly match pixel ratio
            lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) /
                       (img.size(-1) * img.size(-2)))
            mixed_gt_label = lam * one_hot_gt_label + (
                1 - lam) * one_hot_gt_label[index, :]
            return img, mixed_gt_label
        else:
            one_hot_gt_label = F.one_hot(
                gt_label, num_classes=self.num_classes)
            return img, one_hot_gt_label

    def __call__(self, img, gt_label):
        return self.cutmix(img, gt_label)
