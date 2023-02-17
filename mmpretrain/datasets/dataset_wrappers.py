# Copyright (c) OpenMMLab. All rights reserved.
import copy

import numpy as np
from mmengine.dataset import BaseDataset, force_full_init

from mmpretrain.registry import DATASETS


@DATASETS.register_module()
class KFoldDataset:
    """A wrapper of dataset for K-Fold cross-validation.

    K-Fold cross-validation divides all the samples in groups of samples,
    called folds, of almost equal sizes. And we use k-1 of folds to do training
    and use the fold left to do validation.

    Args:
        dataset (:obj:`mmengine.dataset.BaseDataset` | dict): The dataset to be
            divided
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
        if isinstance(dataset, dict):
            self.dataset = DATASETS.build(dataset)
            # Init the dataset wrapper lazily according to the dataset setting.
            lazy_init = dataset.get('lazy_init', False)
        elif isinstance(dataset, BaseDataset):
            self.dataset = dataset
        else:
            raise TypeError(f'Unsupported dataset type {type(dataset)}.')

        self._metainfo = getattr(self.dataset, 'metainfo', {})
        self.fold = fold
        self.num_splits = num_splits
        self.test_mode = test_mode
        self.seed = seed

        self._fully_initialized = False
        if not lazy_init:
            self.full_init()

    @property
    def metainfo(self) -> dict:
        """Get the meta information of ``self.dataset``.

        Returns:
            dict: Meta information of the dataset.
        """
        # Prevent `self._metainfo` from being modified by outside.
        return copy.deepcopy(self._metainfo)

    def full_init(self):
        """fully initialize the dataset."""
        if self._fully_initialized:
            return

        self.dataset.full_init()
        ori_len = len(self.dataset)
        indices = list(range(ori_len))
        if self.seed is not None:
            rng = np.random.default_rng(self.seed)
            rng.shuffle(indices)

        test_start = ori_len * self.fold // self.num_splits
        test_end = ori_len * (self.fold + 1) // self.num_splits
        if self.test_mode:
            indices = indices[test_start:test_end]
        else:
            indices = indices[:test_start] + indices[test_end:]

        self._ori_indices = indices
        self.dataset = self.dataset.get_subset(indices)

        self._fully_initialized = True

    @force_full_init
    def _get_ori_dataset_idx(self, idx: int) -> int:
        """Convert global idx to local index.

        Args:
            idx (int): Global index of ``KFoldDataset``.

        Returns:
            int: The original index in the whole dataset.
        """
        return self._ori_indices[idx]

    @force_full_init
    def get_data_info(self, idx: int) -> dict:
        """Get annotation by index.

        Args:
            idx (int): Global index of ``KFoldDataset``.

        Returns:
            dict: The idx-th annotation of the datasets.
        """
        return self.dataset.get_data_info(idx)

    @force_full_init
    def __len__(self):
        return len(self.dataset)

    @force_full_init
    def __getitem__(self, idx):
        return self.dataset[idx]

    @force_full_init
    def get_cat_ids(self, idx):
        return self.dataset.get_cat_ids(idx)

    @force_full_init
    def get_gt_labels(self):
        return self.dataset.get_gt_labels()

    @property
    def CLASSES(self):
        """Return all categories names."""
        return self._metainfo.get('classes', None)

    @property
    def class_to_idx(self):
        """Map mapping class name to class index.

        Returns:
            dict: mapping from class name to class index.
        """

        return {cat: i for i, cat in enumerate(self.CLASSES)}

    def __repr__(self):
        """Print the basic information of the dataset.

        Returns:
            str: Formatted string.
        """
        head = 'Dataset ' + self.__class__.__name__
        body = []
        type_ = 'test' if self.test_mode else 'training'
        body.append(f'Type: \t{type_}')
        body.append(f'Seed: \t{self.seed}')

        def ordinal(n):
            # Copy from https://codegolf.stackexchange.com/a/74047
            suffix = 'tsnrhtdd'[(n // 10 % 10 != 1) * (n % 10 < 4) * n % 10::4]
            return f'{n}{suffix}'

        body.append(
            f'Fold: \t{ordinal(self.fold+1)} of {self.num_splits}-fold')
        if self._fully_initialized:
            body.append(f'Number of samples: \t{self.__len__()}')
        else:
            body.append("Haven't been initialized")

        if self.CLASSES is not None:
            body.append(f'Number of categories: \t{len(self.CLASSES)}')
        else:
            body.append('The `CLASSES` meta info is not set.')

        body.append(
            f'Original dataset type:\t{self.dataset.__class__.__name__}')

        lines = [head] + [' ' * 4 + line for line in body]
        return '\n'.join(lines)
