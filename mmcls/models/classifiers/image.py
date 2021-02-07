import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.beta import Beta

from ..builder import CLASSIFIERS, build_backbone, build_head, build_neck
from .base import BaseClassifier


@CLASSIFIERS.register_module()
class ImageClassifier(BaseClassifier):

    def __init__(self,
                 backbone,
                 neck=None,
                 head=None,
                 pretrained=None,
                 mixup=0.0):
        super(ImageClassifier, self).__init__()
        self.mixup = mixup
        self.Beta = None
        self.num_classes = 0
        if mixup > 0:
            self.Beta = Beta(self.mixup, self.mixup)

        self.backbone = build_backbone(backbone)

        if neck is not None:
            self.neck = build_neck(neck)

        if head is not None:
            self.num_classes = head['num_classes']
            self.head = build_head(head)

        self.init_weights(pretrained=pretrained)

    def init_weights(self, pretrained=None):
        super(ImageClassifier, self).init_weights(pretrained)
        self.backbone.init_weights(pretrained=pretrained)
        if self.with_neck:
            if isinstance(self.neck, nn.Sequential):
                for m in self.neck:
                    m.init_weights()
            else:
                self.neck.init_weights()
        if self.with_head:
            self.head.init_weights()

    def extract_feat(self, img):
        """Directly extract features from the backbone + neck
        """
        x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)
        return x

    def forward_train(self, img, gt_label, **kwargs):
        """Forward computation during training.

        Args:
            img (Tensor): of shape (N, C, H, W) encoding input images.
                Typically these should be mean centered and std scaled.

            gt_label (Tensor): It should be of shape (N, 1) encoding the
                ground-truth label of input images for single label task. It
                shoulf be of shape (N, C) encoding the ground-truth label
                of input images for multi-labels task.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        if self.mixup > 0:
            img, gt_label = self._mixup(img, gt_label)

        x = self.extract_feat(img)

        losses = dict()
        loss = self.head.forward_train(x, gt_label)
        losses.update(loss)

        return losses

    def simple_test(self, img, img_metas):
        """Test without augmentation."""
        x = self.extract_feat(img)
        return self.head.simple_test(x)

    def _mixup(self, img, gt_label):
        lam = self.Beta.sample()
        batch_size = img.size(0)
        index = torch.randperm(batch_size)
        one_hot_gt_label = F.one_hot(gt_label, num_classes=self.num_classes)

        mixed_img = lam * img + (1 - lam) * img[index, :]
        mixed_gt_label = lam * one_hot_gt_label + (
            1 - lam) * one_hot_gt_label[index, :]
        return mixed_img, mixed_gt_label
