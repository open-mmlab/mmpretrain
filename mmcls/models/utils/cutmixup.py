import numpy as np

from .cutmix import BatchCutMixLayer
from .mixup import BatchMixupLayer


class CutMixUp(object):

    def __init__(self,
                 mixup_alpha=1.,
                 cutmix_alpha=0.,
                 cutmix_minmax=None,
                 prob=1.0,
                 switch_prob=0.5,
                 mode='batch',
                 correct_lam=True,
                 num_classes=1000):
        """Mixup/Cutmix that applies different params to each element or whole
        batch.

        Args:
            mixup_alpha (float): mixup alpha value, mixup is active if > 0.
            cutmix_alpha (float): cutmix alpha value,
                cutmix is active if > 0.
            cutmix_minmax (List[float]): cutmix min/max image ratio,
                cutmix is active and uses this vs alpha if not None.
            prob (float): probability of applying mixup or cutmix per batch
                or element
            switch_prob (float): probability of switching to cutmix
                instead of mixup when both are active
            mode (str): how to apply mixup/cutmix params
                (per 'batch', 'pair' (pair of elements), 'elem' (element))
            correct_lam (bool): apply lambda correction when cutmix bbox
                clipped by image borders
            num_classes (int): number of classes for target
        """

        super(CutMixUp, self).__init__()

        self.mixup_alpha = mixup_alpha
        self.cutmix_alpha = cutmix_alpha
        self.cutmix_minmax = cutmix_minmax
        if self.cutmix_minmax is not None:
            assert len(self.cutmix_minmax) == 2
            # force cutmix alpha == 1.0 when minmax active
            # to keep logic simple & safe
            self.cutmix_alpha = 1.0
        self.mix_prob = prob
        self.switch_prob = switch_prob
        self.num_classes = num_classes
        self.mode = mode
        assert self.mode in ['batch', 'pair', 'elem'], \
            'CutMixUp mode should be one of "batch", "pair" and "elem", ' \
            'but got "{}"'.format(self.mode)
        # correct lambda based on clipped area for cutmix
        self.correct_lam = correct_lam
        # set to false to disable mixing (intended tp be set by train loop)
        self.mixup_enabled = True

    def _mix_batch(self, img, gt_label):
        if not self.mixup_enabled:
            return img, gt_label
        if self.mixup_alpha > 0.:
            mixup = BatchMixupLayer(self.mixup_alpha, self.num_classes,
                                    self.mix_prob)
        if self.cutmix_alpha > 0.:
            cutmix = BatchCutMixLayer(self.cutmix_alpha, self.num_classes,
                                      self.mix_prob, self.cutmix_minmax,
                                      self.correct_lam)
        if self.mixup_alpha > 0. and self.cutmix_alpha > 0.:
            use_cutmix = np.random.rand() < self.switch_prob
            img, gt_label = cutmix(img, gt_label) if use_cutmix else \
                mixup(img, gt_label)
        elif self.mixup_alpha > 0.:
            img, gt_label = mixup(img, gt_label)
        elif self.cutmix_alpha > 0.:
            img, gt_label = cutmix(img, gt_label)
        else:
            assert False, 'One of mixup_alpha > 0., cutmix_alpha > 0.,' \
                          'cutmix_minmax not None should be true.'
        return img, gt_label

    def __call__(self, img, gt_label):
        assert len(img) % 2 == 0, 'Batch size should be even when using this'
        if self.mode == 'batch':
            return self._mix_batch(img, gt_label)
        else:
            raise NotImplementedError('mode "{}" not implemented'.format(
                self.mode))
