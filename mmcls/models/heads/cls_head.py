from ..builder import HEADS, build_loss
from .base_head import BaseHead


@HEADS.register_module()
class ClsHead(BaseHead):
    """classification head.

    Args:
        loss (dict): Config of classification loss.
    """  # noqa: W605

    def __init__(self, loss=dict(type='CrossEntropyLoss', loss_weight=1.0)):
        super(ClsHead, self).__init__()

        assert isinstance(loss, dict)
        self.compute_loss = build_loss(loss)

    def loss(self, cls_score, gt_label):
        num_samples = len(cls_score)
        losses = dict()
        # compute loss
        loss = self.compute_loss(cls_score, gt_label, avg_factor=num_samples)
        losses['loss'] = loss
        return losses

    def forward_train(self, cls_score, gt_label):
        losses = self.loss(cls_score, gt_label)
        return losses
