# Copyright (c) OpenMMLab. All rights reserved.
from abc import ABCMeta, abstractmethod

import numpy as np
import torch

from .builder import AUGMENT
from .utils import one_hot_encoding


class BaseCutMixLayer(object, metaclass=ABCMeta):
    """Base class for CutMixLayer.

    Args:
        alpha (float): Parameters for Beta distribution. Positive(>0)
        num_classes (int): The number of classes
        prob (float): MixUp probability. It should be in range [0, 1].
            Default to 1.0
        cutmix_minmax (List[float], optional): cutmix min/max image ratio.
            (as percent of image size). When cutmix_minmax is not None, we
            generate cutmix bounding-box using cutmix_minmax instead of alpha
        correct_lam (bool): Whether to apply lambda correction when cutmix bbox
            clipped by image borders. Default to True
    """

    def __init__(self,
                 alpha,
                 num_classes,
                 prob=1.0,
                 cutmix_minmax=None,
                 correct_lam=True):
        super(BaseCutMixLayer, self).__init__()

        assert isinstance(alpha, float) and alpha > 0
        assert isinstance(num_classes, int)
        assert isinstance(prob, float) and 0.0 <= prob <= 1.0

        self.alpha = alpha
        self.num_classes = num_classes
        self.prob = prob
        self.cutmix_minmax = cutmix_minmax
        self.correct_lam = correct_lam

    def rand_bbox_minmax(self, img_shape, count=None):
        """Min-Max CutMix bounding-box Inspired by Darknet cutmix
        implementation. It generates a random rectangular bbox based on min/max
        percent values applied to each dimension of the input image.

        Typical defaults for minmax are usually in the  .2-.3 for min and
        .8-.9 range for max.

        Args:
            img_shape (tuple): Image shape as tuple
            count (int, optional): Number of bbox to generate. Default to None
        """
        assert len(self.cutmix_minmax) == 2
        img_h, img_w = img_shape[-2:]
        cut_h = np.random.randint(
            int(img_h * self.cutmix_minmax[0]),
            int(img_h * self.cutmix_minmax[1]),
            size=count)
        cut_w = np.random.randint(
            int(img_w * self.cutmix_minmax[0]),
            int(img_w * self.cutmix_minmax[1]),
            size=count)
        yl = np.random.randint(0, img_h - cut_h, size=count)
        xl = np.random.randint(0, img_w - cut_w, size=count)
        yu = yl + cut_h
        xu = xl + cut_w
        return yl, yu, xl, xu

    def rand_bbox(self, img_shape, lam, margin=0., count=None):
        """Standard CutMix bounding-box that generates a random square bbox
        based on lambda value. This implementation includes support for
        enforcing a border margin as percent of bbox dimensions.

        Args:
            img_shape (tuple): Image shape as tuple
            lam (float): Cutmix lambda value
            margin (float): Percentage of bbox dimension to enforce as margin
                (reduce amount of box outside image). Default to 0.
            count (int, optional): Number of bbox to generate. Default to None
        """
        ratio = np.sqrt(1 - lam)
        img_h, img_w = img_shape[-2:]
        cut_h, cut_w = int(img_h * ratio), int(img_w * ratio)
        margin_y, margin_x = int(margin * cut_h), int(margin * cut_w)
        cy = np.random.randint(0 + margin_y, img_h - margin_y, size=count)
        cx = np.random.randint(0 + margin_x, img_w - margin_x, size=count)
        yl = np.clip(cy - cut_h // 2, 0, img_h)
        yh = np.clip(cy + cut_h // 2, 0, img_h)
        xl = np.clip(cx - cut_w // 2, 0, img_w)
        xh = np.clip(cx + cut_w // 2, 0, img_w)
        return yl, yh, xl, xh

    def cutmix_bbox_and_lam(self, img_shape, lam, count=None):
        """Generate bbox and apply lambda correction.

        Args:
            img_shape (tuple): Image shape as tuple
            lam (float): Cutmix lambda value
            count (int, optional): Number of bbox to generate. Default to None
        """
        if self.cutmix_minmax is not None:
            yl, yu, xl, xu = self.rand_bbox_minmax(img_shape, count=count)
        else:
            yl, yu, xl, xu = self.rand_bbox(img_shape, lam, count=count)
        if self.correct_lam or self.cutmix_minmax is not None:
            bbox_area = (yu - yl) * (xu - xl)
            lam = 1. - bbox_area / float(img_shape[-2] * img_shape[-1])
        return (yl, yu, xl, xu), lam

    @abstractmethod
    def cutmix(self, imgs, gt_label):
        pass


@AUGMENT.register_module(name='BatchCutMix')
class BatchCutMixLayer(BaseCutMixLayer):
    r"""CutMix layer for a batch of data.

    CutMix is a method to improve the network's generalization capability. It's
    proposed in `CutMix: Regularization Strategy to Train Strong Classifiers
    with Localizable Features <https://arxiv.org/abs/1905.04899>`

    With this method, patches are cut and pasted among training images where
    the ground truth labels are also mixed proportionally to the area of the
    patches.

    Args:
        alpha (float): Parameters for Beta distribution to generate the
            mixing ratio. It should be a positive number. More details
            can be found in :class:`BatchMixupLayer`.
        num_classes (int): The number of classes
        prob (float): The probability to execute cutmix. It should be in
            range [0, 1]. Defaults to 1.0.
        cutmix_minmax (List[float], optional): The min/max area ratio of the
            patches. If not None, the bounding-box of patches is uniform
            sampled within this ratio range, and the ``alpha`` will be ignored.
            Otherwise, the bounding-box is generated according to the
            ``alpha``. Defaults to None.
        correct_lam (bool): Whether to apply lambda correction when cutmix bbox
            clipped by image borders. Defaults to True.

    Note:
        If the ``cutmix_minmax`` is None, how to generate the bounding-box of
        patches according to the ``alpha``?

        First, generate a :math:`\lambda`, details can be found in
        :class:`BatchMixupLayer`. And then, the area ratio of the bounding-box
        is calculated by:

        .. math::
            \text{ratio} = \sqrt{1-\lambda}
    """

    def __init__(self, *args, **kwargs):
        super(BatchCutMixLayer, self).__init__(*args, **kwargs)

    def cutmix(self, img, gt_label):
        one_hot_gt_label = one_hot_encoding(gt_label, self.num_classes)
        lam = np.random.beta(self.alpha, self.alpha)
        batch_size = img.size(0)
        index = torch.randperm(batch_size)

        (bby1, bby2, bbx1,
         bbx2), lam = self.cutmix_bbox_and_lam(img.shape, lam)
        img[:, :, bby1:bby2, bbx1:bbx2] = \
            img[index, :, bby1:bby2, bbx1:bbx2]
        mixed_gt_label = lam * one_hot_gt_label + (
            1 - lam) * one_hot_gt_label[index, :]
        return img, mixed_gt_label

    def __call__(self, img, gt_label):
        return self.cutmix(img, gt_label)
