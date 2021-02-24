import bisect
import math
from collections import defaultdict

import numpy as np
from torch.utils.data.dataset import ConcatDataset as _ConcatDataset

from .builder import DATASETS


@DATASETS.register_module()
class ConcatDataset(_ConcatDataset):
    """A wrapper of concatenated dataset.

    Same as :obj:`torch.utils.data.dataset.ConcatDataset`, but
    add `get_cat_ids` function.

    Args:
        datasets (list[:obj:`Dataset`]): A list of datasets.
    """

    def __init__(self, datasets):
        super(ConcatDataset, self).__init__(datasets)
        self.CLASSES = datasets[0].CLASSES

    def get_cat_ids(self, idx):
        if idx < 0:
            if -idx > len(self):
                raise ValueError(
                    'absolute value of index should not exceed dataset length')
            idx = len(self) + idx
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        return self.datasets[dataset_idx].get_cat_ids(sample_idx)


@DATASETS.register_module()
class RepeatDataset(object):
    """A wrapper of repeated dataset.

    The length of repeated dataset will be `times` larger than the original
    dataset. This is useful when the data loading time is long but the dataset
    is small. Using RepeatDataset can reduce the data loading time between
    epochs.

    Args:
        dataset (:obj:`Dataset`): The dataset to be repeated.
        times (int): Repeat times.
    """

    def __init__(self, dataset, times):
        self.dataset = dataset
        self.times = times
        self.CLASSES = dataset.CLASSES

        self._ori_len = len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx % self._ori_len]

    def get_cat_ids(self, idx):
        return self.dataset.get_cat_ids(idx % self._ori_len)

    def __len__(self):
        return self.times * self._ori_len


# Modified from https://github.com/facebookresearch/detectron2/blob/41d475b75a230221e21d9cac5d69655e3415e3a4/detectron2/data/samplers/distributed_sampler.py#L57 # noqa
@DATASETS.register_module()
class ClassBalancedDataset(object):
    """A wrapper of repeated dataset with repeat factor.

    Suitable for training on class imbalanced datasets like LVIS. Following
    the sampling strategy in [1], in each epoch, an image may appear multiple
    times based on its "repeat factor".
    The repeat factor for an image is a function of the frequency the rarest
    category labeled in that image. The "frequency of category c" in [0, 1]
    is defined by the fraction of images in the training set (without repeats)
    in which category c appears.
    The dataset needs to instantiate :func:`self.get_cat_ids(idx)` to support
    ClassBalancedDataset.
    The repeat factor is computed as followed.
    1. For each category c, compute the fraction # of images
        that contain it: f(c)
    2. For each category c, compute the category-level repeat factor:
        r(c) = max(1, sqrt(t/f(c)))
    3. For each image I and its labels L(I), compute the image-level repeat
    factor:
        r(I) = max_{c in L(I)} r(c)

    References:
        .. [1]  https://arxiv.org/pdf/1908.03195.pdf

    Args:
        dataset (:obj:`CustomDataset`): The dataset to be repeated.
        oversample_thr (float): frequency threshold below which data is
            repeated. For categories with `f_c` >= `oversample_thr`, there is
            no oversampling. For categories with `f_c` < `oversample_thr`, the
            degree of oversampling following the square-root inverse frequency
            heuristic above.
    """

    def __init__(self, dataset, oversample_thr):
        self.dataset = dataset
        self.oversample_thr = oversample_thr
        self.CLASSES = dataset.CLASSES

        repeat_factors = self._get_repeat_factors(dataset, oversample_thr)
        repeat_indices = []
        for dataset_index, repeat_factor in enumerate(repeat_factors):
            repeat_indices.extend([dataset_index] * math.ceil(repeat_factor))
        self.repeat_indices = repeat_indices

        flags = []
        if hasattr(self.dataset, 'flag'):
            for flag, repeat_factor in zip(self.dataset.flag, repeat_factors):
                flags.extend([flag] * int(math.ceil(repeat_factor)))
            assert len(flags) == len(repeat_indices)
        self.flag = np.asarray(flags, dtype=np.uint8)

    def _get_repeat_factors(self, dataset, repeat_thr):
        # 1. For each category c, compute the fraction # of images
        #   that contain it: f(c)
        category_freq = defaultdict(int)
        num_images = len(dataset)
        for idx in range(num_images):
            cat_ids = set(self.dataset.get_cat_ids(idx))
            for cat_id in cat_ids:
                category_freq[cat_id] += 1
        for k, v in category_freq.items():
            assert v > 0, f'caterogy {k} does not contain any images'
            category_freq[k] = v / num_images

        # 2. For each category c, compute the category-level repeat factor:
        #    r(c) = max(1, sqrt(t/f(c)))
        category_repeat = {
            cat_id: max(1.0, math.sqrt(repeat_thr / cat_freq))
            for cat_id, cat_freq in category_freq.items()
        }

        # 3. For each image I and its labels L(I), compute the image-level
        # repeat factor:
        #    r(I) = max_{c in L(I)} r(c)
        repeat_factors = []
        for idx in range(num_images):
            cat_ids = set(self.dataset.get_cat_ids(idx))
            repeat_factor = max(
                {category_repeat[cat_id]
                 for cat_id in cat_ids})
            repeat_factors.append(repeat_factor)

        return repeat_factors

    def __getitem__(self, idx):
        ori_index = self.repeat_indices[idx]
        return self.dataset[ori_index]

    def __len__(self):
        return len(self.repeat_indices)
