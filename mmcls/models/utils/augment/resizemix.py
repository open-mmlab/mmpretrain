# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import torch
import torch.nn.functional as F

from mmcls.models.utils.augment.builder import AUGMENT
from .cutmix import BatchCutMixLayer
from .utils import one_hot_encoding


@AUGMENT.register_module(name='BatchResizeMix')
class BatchResizeMixLayer(BatchCutMixLayer):
    r"""ResizeMix Random Paste layer for a batch of data.

    The ResizeMix will resize an image to a small patch and paste it on another
    image. It's proposed in `ResizeMix: Mixing Data with Preserved Object
    Information and True Labels <https://arxiv.org/abs/2012.11101>`_

    Args:
        alpha (float): Parameters for Beta distribution to generate the
            mixing ratio. It should be a positive number. More details
            can be found in :class:`BatchMixupLayer`.
        num_classes (int): The number of classes.
        lam_min(float): The minimum value of lam. Defaults to 0.1.
        lam_max(float): The maximum value of lam. Defaults to 0.8.
        interpolation (str): algorithm used for upsampling:
            'nearest' | 'linear' | 'bilinear' | 'bicubic' | 'trilinear' |
            'area'. Default to 'bilinear'.
        prob (float): The probability to execute resizemix. It should be in
            range [0, 1]. Defaults to 1.0.
        cutmix_minmax (List[float], optional): The min/max area ratio of the
            patches. If not None, the bounding-box of patches is uniform
            sampled within this ratio range, and the ``alpha`` will be ignored.
            Otherwise, the bounding-box is generated according to the
            ``alpha``. Defaults to None.
        correct_lam (bool): Whether to apply lambda correction when cutmix bbox
            clipped by image borders. Defaults to True
        **kwargs: Any other parameters accpeted by :class:`BatchCutMixLayer`.

    Note:
        The :math:`\lambda` (``lam``) is the mixing ratio. It's a random
        variable which follows :math:`Beta(\alpha, \alpha)` and is mapped
        to the range [``lam_min``, ``lam_max``].

        .. math::
            \lambda = \frac{Beta(\alpha, \alpha)}
            {\lambda_{max} - \lambda_{min}} + \lambda_{min}

        And the resize ratio of source images is calculated by :math:`\lambda`:

        .. math::
            \text{ratio} = \sqrt{1-\lambda}
    """

    def __init__(self,
                 alpha,
                 num_classes,
                 lam_min: float = 0.1,
                 lam_max: float = 0.8,
                 interpolation='bilinear',
                 prob=1.0,
                 cutmix_minmax=None,
                 correct_lam=True,
                 **kwargs):
        super(BatchResizeMixLayer, self).__init__(
            alpha=alpha,
            num_classes=num_classes,
            prob=prob,
            cutmix_minmax=cutmix_minmax,
            correct_lam=correct_lam,
            **kwargs)
        self.lam_min = lam_min
        self.lam_max = lam_max
        self.interpolation = interpolation

    def cutmix(self, img, gt_label):
        one_hot_gt_label = one_hot_encoding(gt_label, self.num_classes)

        lam = np.random.beta(self.alpha, self.alpha)
        lam = lam * (self.lam_max - self.lam_min) + self.lam_min
        batch_size = img.size(0)
        index = torch.randperm(batch_size)

        (bby1, bby2, bbx1,
         bbx2), lam = self.cutmix_bbox_and_lam(img.shape, lam)

        img[:, :, bby1:bby2, bbx1:bbx2] = F.interpolate(
            img[index],
            size=(bby2 - bby1, bbx2 - bbx1),
            mode=self.interpolation)
        mixed_gt_label = lam * one_hot_gt_label + (
            1 - lam) * one_hot_gt_label[index, :]
        return img, mixed_gt_label
