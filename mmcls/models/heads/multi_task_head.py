from ..builder import HEADS, build_head
from .cls_head import ClsHead


@HEADS.register_module()
class MultiTaskClsHead(ClsHead):
    """Multi task head.

    Args:
        heads (dict): Sub heads to use,
        the key will be use to rename the loss components
        base_head (dict): Default dict config for heads
        default: None
    """

    def __init__(self, heads, base_head=None, **kwargs):
        super(MultiTaskClsHead, self).__init__(**kwargs)

        assert isinstance(heads, (dict))

        head_values = heads.values()
        self.names = list(heads.keys())
        self.heads = []

        for (index, head) in enumerate(head_values):
            if base_head is not None:
                head.update(base_head)
            module_head = build_head(head)
            self.heads.append(module_head)
            self.add_module(self.names[index], module_head)

    def forward_train(self, x, gt_label, **kwargs):

        losses = dict()
        for (index, head) in enumerate(self.heads):
            head_loss = head.forward_train(x, gt_label[index], **kwargs)
            losses[f'loss_{self.names[index]}'] = head_loss['loss']
        return losses

    def pre_logits(self, x):
        if isinstance(x, tuple):
            x = x[-1]
        return x

    def simple_test(self, x, post_process=True, **kwargs):
        """Inference without augmentation.

        Args:
            x (tuple[Tensor]): The input features
            will be forwarded to each head
        Returns:
            list[tuple]: The inference results.
                the output is a list, one item per batch element,
                each item contains a tuple, containing the results of each head
        """
        assert post_process, 'post_process=False\
         is not implemented for MultiTaskClsHead.simple_test'

        return list(
            zip(*[
                head.simple_test(x, post_process=post_process, **kwargs)
                for head in self.heads
            ]))
