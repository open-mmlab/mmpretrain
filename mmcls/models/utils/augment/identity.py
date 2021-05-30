import torch.nn.functional as F

from .builder import AUGMENT


@AUGMENT.register_module(name='Identity')
class Identity(object):

    def __init__(self, num_classes, prob=0.5):
        super(Identity, self).__init__()

        assert isinstance(num_classes, int)
        assert isinstance(prob, float) and 0.0 <= prob <= 1.0

        self.num_classes = num_classes
        self.prob = prob

    def one_hot(self, gt_label):
        return F.one_hot(gt_label, num_classes=self.num_classes)

    def __call__(self, img, gt_label):
        return img, self.one_hot(gt_label)
