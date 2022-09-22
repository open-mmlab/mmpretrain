# Copyright (c) OpenMMLab. All rights reserved.
import bisect
import math
from collections import defaultdict

import numpy as np
from mmcv.utils import print_log
from torch.utils.data.dataset import ConcatDataset as _ConcatDataset

from .builder import DATASETS


@DATASETS.register_module()
class ConcatDataset(_ConcatDataset):
    """A wrapper of concatenated dataset.

    Same as :obj:`torch.utils.data.dataset.ConcatDataset`, but
    add `get_cat_ids` function.

    Args:
        datasets (list[:obj:`BaseDataset`]): A list of datasets.
        separate_eval (bool): Whether to evaluate the results
            separately if it is used as validation dataset.
            Defaults to True.
    """

    def __init__(self, datasets, separate_eval=True):
        super(ConcatDataset, self).__init__(datasets)
        self.separate_eval = separate_eval

        self.CLASSES = datasets[0].CLASSES

        if not separate_eval:
            if len(set([type(ds) for ds in datasets])) != 1:
                raise NotImplementedError(
                    'To evaluate a concat dataset non-separately, '
                    'all the datasets should have same types')

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

    def evaluate(self, results, *args, indices=None, logger=None, **kwargs):
        """Evaluate the results.

        Args:
            results (list[list | tuple]): Testing results of the dataset.
            indices (list, optional): The indices of samples corresponding to
                the results. It's unavailable on ConcatDataset.
                Defaults to None.
            logger (logging.Logger | str, optional): Logger used for printing
                related information during evaluation. Defaults to None.

        Returns:
            dict[str: float]: AP results of the total dataset or each separate
            dataset if `self.separate_eval=True`.
        """
        if indices is not None:
            raise NotImplementedError(
                'Use indices to evaluate speific samples in a ConcatDataset '
                'is not supported by now.')

        assert len(results) == len(self), \
            ('Dataset and results have different sizes: '
             f'{len(self)} v.s. {len(results)}')

        # Check whether all the datasets support evaluation
        for dataset in self.datasets:
            assert hasattr(dataset, 'evaluate'), \
                f"{type(dataset)} haven't implemented the evaluate function."

        if self.separate_eval:
            total_eval_results = dict()
            for dataset_idx, dataset in enumerate(self.datasets):
                start_idx = 0 if dataset_idx == 0 else \
                    self.cumulative_sizes[dataset_idx-1]
                end_idx = self.cumulative_sizes[dataset_idx]

                results_per_dataset = results[start_idx:end_idx]
                print_log(
                    f'Evaluateing dataset-{dataset_idx} with '
                    f'{len(results_per_dataset)} images now',
                    logger=logger)

                eval_results_per_dataset = dataset.evaluate(
                    results_per_dataset, *args, logger=logger, **kwargs)
                for k, v in eval_results_per_dataset.items():
                    total_eval_results.update({f'{dataset_idx}_{k}': v})

            return total_eval_results
        else:
            original_data_infos = self.datasets[0].data_infos
            self.datasets[0].data_infos = sum(
                [dataset.data_infos for dataset in self.datasets], [])
            eval_results = self.datasets[0].evaluate(
                results, logger=logger, **kwargs)
            self.datasets[0].data_infos = original_data_infos
            return eval_results


@DATASETS.register_module()
class RepeatDataset(object):
    """A wrapper of repeated dataset.

    The length of repeated dataset will be `times` larger than the original
    dataset. This is useful when the data loading time is long but the dataset
    is small. Using RepeatDataset can reduce the data loading time between
    epochs.

    Args:
        dataset (:obj:`BaseDataset`): The dataset to be repeated.
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

    def evaluate(self, *args, **kwargs):
        raise NotImplementedError(
            'evaluate results on a repeated dataset is weird. '
            'Please inference and evaluate on the original dataset.')

    def __repr__(self):
        """Print the number of instance number."""
        dataset_type = 'Test' if self.test_mode else 'Train'
        result = (
            f'\n{self.__class__.__name__} ({self.dataset.__class__.__name__}) '
            f'{dataset_type} dataset with total number of samples {len(self)}.'
        )
        return result


# Modified from https://github.com/facebookresearch/detectron2/blob/41d475b75a230221e21d9cac5d69655e3415e3a4/detectron2/data/samplers/distributed_sampler.py#L57 # noqa
@DATASETS.register_module()
class ClassBalancedDataset(object):
    r"""A wrapper of repeated dataset with repeat factor.

    Suitable for training on class imbalanced datasets like LVIS. Following the
    sampling strategy in `this paper`_, in each epoch, an image may appear
    multiple times based on its "repeat factor".

    .. _this paper: https://arxiv.org/pdf/1908.03195.pdf

    The repeat factor for an image is a function of the frequency the rarest
    category labeled in that image. The "frequency of category c" in [0, 1]
    is defined by the fraction of images in the training set (without repeats)
    in which category c appears.

    The dataset needs to implement :func:`self.get_cat_ids` to support
    ClassBalancedDataset.

    The repeat factor is computed as followed.

    1. For each category c, compute the fraction :math:`f(c)` of images that
       contain it.
    2. For each category c, compute the category-level repeat factor

        .. math::
            r(c) = \max(1, \sqrt{\frac{t}{f(c)}})

    3. For each image I and its labels :math:`L(I)`, compute the image-level
       repeat factor

        .. math::
            r(I) = \max_{c \in L(I)} r(c)

    Args:
        dataset (:obj:`BaseDataset`): The dataset to be repeated.
        oversample_thr (float): frequency threshold below which data is
            repeated. For categories with ``f_c`` >= ``oversample_thr``, there
            is no oversampling. For categories with ``f_c`` <
            ``oversample_thr``, the degree of oversampling following the
            square-root inverse frequency heuristic above.
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

    def evaluate(self, *args, **kwargs):
        raise NotImplementedError(
            'evaluate results on a class-balanced dataset is weird. '
            'Please inference and evaluate on the original dataset.')

    def __repr__(self):
        """Print the number of instance number."""
        dataset_type = 'Test' if self.test_mode else 'Train'
        result = (
            f'\n{self.__class__.__name__} ({self.dataset.__class__.__name__}) '
            f'{dataset_type} dataset with total number of samples {len(self)}.'
        )
        return result


@DATASETS.register_module()
class KFoldDataset:
    """A wrapper of dataset for K-Fold cross-validation.

    K-Fold cross-validation divides all the samples in groups of samples,
    called folds, of almost equal sizes. And we use k-1 of folds to do training
    and use the fold left to do validation.

    Args:
        dataset (:obj:`BaseDataset`): The dataset to be divided.
        fold (int): The fold used to do validation. Defaults to 0.
        num_splits (int): The number of all folds. Defaults to 5.
        test_mode (bool): Use the training dataset or validation dataset.
            Defaults to False.
        seed (int, optional): The seed to shuffle the dataset before splitting.
            If None, not shuffle the dataset. Defaults to None.
    """

    def __init__(self,
                 dataset,
                 fold=0,
                 num_splits=5,
                 test_mode=False,
                 seed=None):
        self.dataset = dataset
        self.CLASSES = dataset.CLASSES
        self.test_mode = test_mode
        self.num_splits = num_splits

        length = len(dataset)
        indices = list(range(length))
        if isinstance(seed, int):
            rng = np.random.default_rng(seed)
            rng.shuffle(indices)

        test_start = length * fold // num_splits
        test_end = length * (fold + 1) // num_splits
        if test_mode:
            self.indices = indices[test_start:test_end]
        else:
            self.indices = indices[:test_start] + indices[test_end:]

    def get_cat_ids(self, idx):
        return self.dataset.get_cat_ids(self.indices[idx])

    def get_gt_labels(self):
        dataset_gt_labels = self.dataset.get_gt_labels()
        gt_labels = np.array([dataset_gt_labels[idx] for idx in self.indices])
        return gt_labels

    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]

    def __len__(self):
        return len(self.indices)

    def evaluate(self, *args, **kwargs):
        kwargs['indices'] = self.indices
        return self.dataset.evaluate(*args, **kwargs)
