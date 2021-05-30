import torch.nn.functional as F

from .base_augment import BaseAugment
from .builder import AUGMENT


@AUGMENT.register_module(name='Identity')
class Identity(BaseAugment):

    def __init__(self, *args, **kwargs):
        super(Identity, self).__init__(*args, **kwargs)

    def one_hot(self, gt_label):
        return F.one_hot(gt_label, num_classes=self.num_classes)

    def __call__(self, img, gt_label):
        return img, self.one_hot(gt_label)
