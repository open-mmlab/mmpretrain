# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np

from mmcls.registry import DATASETS


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
