# Copyright (c) OpenMMLab. All rights reserved.
import random
from abc import ABCMeta

import numpy as np
import torch
import torch.nn.functional as F

from .builder import AUGMENT


@AUGMENT.register_module(name='BatchCutMixup')
class BatchCutMixupLayer(object, metaclass=ABCMeta):

    def __init__(self,
                 alpha_cutmix,
                 alpha_mixup,
                 num_classes,
                 prob_cutmix=1.0,
                 prob_mixup=0.5,
                 prob=1.0,
                 cutmix_minmax=None,
                 correct_lam=True):
        super(BatchCutMixupLayer, self).__init__()

        assert isinstance(alpha_cutmix, float) and alpha_cutmix > 0
        assert isinstance(num_classes, int)
        assert isinstance(prob_cutmix, float) and 0.0 <= prob_cutmix <= 1.0

        self.alpha_cutmix = alpha_cutmix
        self.alpha_mixup = alpha_mixup
        self.num_classes = num_classes
        self.prob_cutmix = prob_cutmix
        self.prob_mixup = prob_mixup
        self.cutmix_minmax = cutmix_minmax
        self.correct_lam = correct_lam
        self.prob = prob

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

    def cutmix(self, img, one_hot_gt_label):
        lam = np.random.beta(self.alpha_cutmix, self.alpha_cutmix)
        batch_size = img.size(0)
        index = torch.randperm(batch_size)

        (bby1, bby2, bbx1,
         bbx2), lam = self.cutmix_bbox_and_lam(img.shape, lam)
        img[:, :, bby1:bby2, bbx1:bbx2] = \
            img[index, :, bby1:bby2, bbx1:bbx2]
        mixed_gt_label = lam * one_hot_gt_label + (
            1 - lam) * one_hot_gt_label[index, :]
        return img, mixed_gt_label

    def mixup(self, img, one_hot_gt_label):
        lam = np.random.beta(self.alpha_mixup, self.alpha_mixup)
        batch_size = img.size(0)
        index = torch.randperm(batch_size)

        mixed_img = lam * img + (1 - lam) * img[index, :]
        mixed_gt_label = lam * one_hot_gt_label + (
            1 - lam) * one_hot_gt_label[index, :]

        return mixed_img, mixed_gt_label

    def cvt_lbl(self, gt_label):
        one_hot_gt_label = F.one_hot(gt_label, num_classes=self.num_classes)
        return one_hot_gt_label

    def __call__(self, img, gt_label):
        one_hot_gt_label = self.cvt_lbl(gt_label)
        if random.random() < self.prob_cutmix:
            img, one_hot_gt_label = self.cutmix(img, one_hot_gt_label)
        if random.random() < self.prob_mixup:
            img, one_hot_gt_label = self.mixup(img, one_hot_gt_label)
        return img, one_hot_gt_label
