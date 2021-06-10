import warnings

from ..builder import CLASSIFIERS, build_backbone, build_head, build_neck
from ..utils import BatchCutMixLayer, BatchMixupLayer
from .base import BaseClassifier


@CLASSIFIERS.register_module()
class ImageClassifier(BaseClassifier):

    def __init__(self,
                 backbone,
                 neck=None,
                 head=None,
                 pretrained=None,
                 train_cfg=None,
                 init_cfg=None):
        super(ImageClassifier, self).__init__(init_cfg)

        if pretrained is not None:
            warnings.warn('DeprecationWarning: pretrained is a deprecated \
                key, please consider using init_cfg')
            self.init_cfg = dict(type='Pretrained', checkpoint=pretrained)

        self.backbone = build_backbone(backbone)

        if neck is not None:
            self.neck = build_neck(neck)

        if head is not None:
            self.head = build_head(head)

        self.mixup, self.cutmix = None, None
        if train_cfg is not None:
            mixup_cfg = train_cfg.get('mixup', None)
            cutmix_cfg = train_cfg.get('cutmix', None)
            assert mixup_cfg is None or cutmix_cfg is None, \
                'Mixup and CutMix can not be set simultaneously.'
            if mixup_cfg is not None:
                self.mixup = BatchMixupLayer(**mixup_cfg)
            if cutmix_cfg is not None:
                self.cutmix = BatchCutMixLayer(**cutmix_cfg)

    def extract_feat(self, img):
        """Directly extract features from the backbone + neck."""
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
        if self.mixup is not None:
            img, gt_label = self.mixup(img, gt_label)

        if self.cutmix is not None:
            img, gt_label = self.cutmix(img, gt_label)

        x = self.extract_feat(img)

        losses = dict()
        loss = self.head.forward_train(x, gt_label)
        losses.update(loss)

        return losses

    def simple_test(self, img, img_metas):
        """Test without augmentation."""
        x = self.extract_feat(img)
        return self.head.simple_test(x)
