import numpy as np
import torch
from mmcv.utils import digit_version
from torchvision.transforms import Resize

from mmcls.models.utils.augment.builder import AUGMENT
from .cutmix import BatchCutMixLayer
from .utils import one_hot_encoding


@AUGMENT.register_module(name='BatchResizeMix')
class BatchResizeMixLayer(BatchCutMixLayer):
    r"""ResizeMix Random Paste layer for batch ResizeMix.

    The ResizeMix will resize an image to a small patch and paste it on another
    image. More details can be found in `ResizeMix: Mixing Data with Preserved
    Object Information and True Labels <https://arxiv.org/abs/2012.11101>`_

    Args:
        alpha (float): Parameters for Beta distribution. Positive(>0)
        num_classes (int): The number of classes.
        lam_min(float): The minimum value of lam. Defaults to 0.1.
        lam_max(float): The maximum value of lam. Defaults to 0.8.
        prob (float): mix probability. It should be in range [0, 1].
            Default to 1.0.
        cutmix_minmax (List[float], optional): cutmix min/max image ratio.
            (as percent of image size). When cutmix_minmax is not None, we
            generate cutmix bounding-box using cutmix_minmax instead of alpha
        correct_lam (bool): Whether to apply lambda correction when cutmix bbox
            clipped by image borders. Default to True
        **kwargs: Any other parameters accpeted by :class:`BatchCutMixLayer`.

    Note:
        The :math:`\lambda` (``lam``) is the mixing ratio. It's a random
        variable which follows :math:`Beta(\alpha, \alpha)` and is mapped
        to the range [``lam_min``, ``lam_max``].

        .. math::
            \lambda = \frac{Beta(\alpha, \alpha)}{\lambda_{max} - \lambda_{min}} + 
            \lambda_{min}

        And the resize ratio of source images is calculated by :math:`\lambda`:

        .. math::
            \text{ratio} = \sqrt{1-lam}
    """
    """

    def __init__(self,
                 alpha,
                 num_classes,
                 lam_min: float = 0.1,
                 lam_max: float = 0.8,
                 prob=1.0,
                 cutmix_minmax=None,
                 correct_lam=True,
                 **kwargs):
        if digit_version(torch.__version__) < digit_version('1.7.0'):
            raise RuntimeError('torchvision.transforms.Resize is not available'
                               'with Tensor before 1.7.0')
        super(BatchResizeMixLayer, self).__init__(
            alpha=alpha,
            num_classes=num_classes,
            prob=prob,
            cutmix_minmax=cutmix_minmax,
            correct_lam=correct_lam,
            **kwargs)
        self.lam_min = lam_min
        self.lam_max = lam_max

    def cutmix(self, img, gt_label):
        one_hot_gt_label = one_hot_encoding(gt_label, self.num_classes)

        lam = np.random.beta(self.alpha, self.alpha)
        lam = lam * (self.lam_max - self.lam_min) + self.lam_min
        batch_size = img.size(0)
        index = torch.randperm(batch_size)

        (bby1, bby2, bbx1,
         bbx2), lam = self.cutmix_bbox_and_lam(img.shape, lam)

        resize_transform = Resize((bby2 - bby1, bbx2 - bbx1))
        img[:, :, bby1:bby2, bbx1:bbx2] = resize_transform(img[index])
        mixed_gt_label = lam * one_hot_gt_label + (
            1 - lam) * one_hot_gt_label[index, :]
        return img, mixed_gt_label
