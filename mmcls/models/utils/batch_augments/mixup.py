# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Optional, Tuple

import numpy as np
import torch
from mmengine.structures import LabelData

from mmcls.registry import BATCH_AUGMENTS
from mmcls.structures import ClsDataSample


@BATCH_AUGMENTS.register_module()
class Mixup:
    r"""Mixup batch augmentation.

    Mixup is a method to reduces the memorization of corrupt labels and
    increases the robustness to adversarial examples. It's proposed in
    `mixup: Beyond Empirical Risk Minimization
    <https://arxiv.org/abs/1710.09412>`_

    Args:
        alpha (float): Parameters for Beta distribution to generate the
            mixing ratio. It should be a positive number. More details
            are in the note.
        num_classes (int, optional): The number of classes. If not specified,
            will try to get it from data samples during training.
            Defaults to None.

    Note:
        The :math:`\alpha` (``alpha``) determines a random distribution
        :math:`Beta(\alpha, \alpha)`. For each batch of data, we sample
        a mixing ratio (marked as :math:`\lambda`, ``lam``) from the random
        distribution.
    """

    def __init__(self, alpha: float, num_classes: Optional[int] = None):
        assert isinstance(alpha, float) and alpha > 0
        assert isinstance(num_classes, int) or num_classes is None

        self.alpha = alpha
        self.num_classes = num_classes

    def mix(self, batch_inputs: torch.Tensor,
            batch_scores: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Mix the batch inputs and batch one-hot format ground truth.

        Args:
            batch_inputs (Tensor): A batch of images tensor in the shape of
                ``(N, C, H, W)``.
            batch_scores (Tensor): A batch of one-hot format labels in the
                shape of ``(N, num_classes)``.

        Returns:
            Tuple[Tensor, Tensor): The mixed inputs and labels.
        """
        lam = np.random.beta(self.alpha, self.alpha)
        batch_size = batch_inputs.size(0)
        index = torch.randperm(batch_size)

        mixed_inputs = lam * batch_inputs + (1 - lam) * batch_inputs[index, :]
        mixed_scores = lam * batch_scores + (1 - lam) * batch_scores[index, :]

        return mixed_inputs, mixed_scores

    def __call__(self, batch_inputs: torch.Tensor,
                 data_samples: List[ClsDataSample]):
        """Mix the batch inputs and batch data samples."""
        assert data_samples is not None, f'{self.__class__.__name__} ' \
            'requires data_samples. If you only want to inference, please ' \
            'disable it from preprocessing.'

        if self.num_classes is None and 'num_classes' not in data_samples[0]:
            raise RuntimeError(
                'Not specify the `num_classes` and cannot get it from '
                'data samples. Please specify `num_classes` in the '
                f'{self.__class__.__name__}.')
        num_classes = self.num_classes or data_samples[0].get('num_classes')

        batch_score = torch.stack([
            LabelData.label_to_onehot(sample.gt_label.label, num_classes)
            for sample in data_samples
        ])

        mixed_inputs, mixed_score = self.mix(batch_inputs, batch_score)

        for i, sample in enumerate(data_samples):
            sample.set_gt_score(mixed_score[i])

        return mixed_inputs, data_samples
