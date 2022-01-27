import numpy as np
import torch
from torchvision.transforms import Resize

from mmcls.models.utils.augment.builder import AUGMENT
from .cutmix import BatchCutMixLayer
from .utils import one_hot_encoding


@AUGMENT.register_module(name='BatchResizeMix')
class BatchResizeMixLayer(BatchCutMixLayer):
    """ResizeMix Random Paste layer for batch ResizeMix.
    Parameters
    ----------
    paste_min:
        minimum value of lam
    lam_max:
        maximum value of lam
    """

    def __init__(self,
                 lam_min: float = 0.1,
                 lam_max: float = 0.8,
                 *args,
                 **kwargs):
        super(BatchResizeMixLayer, self).__init__(*args, **kwargs)
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
