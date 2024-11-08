# Copyright (c) OpenMMLab. All rights reserved.
import collections
import math
from typing import Iterator

import numpy as np
import torch
from mmengine.dataset import DefaultSampler

from mmpretrain.registry import DATA_SAMPLERS


@DATA_SAMPLERS.register_module()
class BatchBalanceSampler(DefaultSampler):
    """
    refer: https://github.com/KevinMusgrave/pytorch-metric-learning/
        blob/v2.3.0/src/pytorch_metric_learning/samplers/num_per_class_sampler.py


       At every iteration, this will return m samples per class. For example,
    if dataloader's batchsize is 100, and m = 5, then 20 classes with 5 samples
    each will be returned


    Args:
        num_per_class: number of samples per class in a batch

    Examples:
        train_dataloader = dict(
            xxxx,
            sampler=dict(type="BatchBalanceSampler", num_per_class=4),
        )
    """

    def __init__(self, num_per_class, **kwargs) -> None:
        super().__init__(**kwargs)
        self.num_per_class = int(num_per_class)
        self.labels_to_indices = self.get_labels_to_indices(
            self.dataset.get_gt_labels())
        self.labels = list(self.labels_to_indices.keys())  # labels index list
        self.length_of_single_pass = self.num_per_class * len(self.labels)

        self.total_size = len(self.dataset)
        # It must be an integer multiple of length_of_single_pass
        if self.length_of_single_pass < self.total_size:
            self.total_size -= (self.total_size) % (self.length_of_single_pass)
        # The number of samples in this rank
        self.num_samples = math.ceil(
            (self.total_size - self.rank) / self.world_size)

    def __len__(self) -> int:
        """The number of samples in this rank."""
        return self.num_samples

    def __iter__(self) -> Iterator[int]:
        indices = [0] * self.total_size
        i = 0
        num_iters = self.calculate_num_iters()
        for _ in range(num_iters):
            np.random.shuffle(self.labels)
            curr_label_set = self.labels
            for label in curr_label_set:
                # List of all sample indexes corresponding to the current label
                t = self.labels_to_indices[label]
                indices[i:i + self.num_per_class] = self.safe_random_choice(
                    t, size=self.num_per_class)
                i += self.num_per_class
        # subsample
        indices = indices[self.rank:self.total_size:self.world_size]

        return iter(indices)

    def calculate_num_iters(self):
        divisor = self.length_of_single_pass
        return self.total_size // divisor if divisor < self.total_size else 1

    def safe_random_choice(self, input_data, size):
        """Randomly samples without replacement from a sequence.

        It is "safe" because
        if len(input_data) < size, it will randomly sample WITH replacement
        Args:
            input_data is a sequence, like a torch tensor, numpy array,
                            python list, tuple etc
            size is the number of elements to randomly sample from input_data
        Returns:
            An array of size "size", randomly sampled from input_data
        """
        replace = len(input_data) < size
        return np.random.choice(input_data, size=size, replace=replace)

    def get_labels_to_indices(self, labels):
        """Creates labels_to_indices, which is a dictionary mapping each label
        to a numpy array of indices that will be used to index into
        self.dataset.

        {labels_index：Index of samples belonging to the category}

        eg： {
                "0":[1,3,6,8],
                "1":[2,4,5,7],
                "2":[0,9,10]
            }
        """
        if torch.is_tensor(labels):
            labels = labels.cpu().numpy()
        labels_to_indices = collections.defaultdict(list)
        for i, label in enumerate(labels):
            labels_to_indices[label].append(i)
        for k, v in labels_to_indices.items():
            labels_to_indices[k] = np.array(v, dtype=int)
        return labels_to_indices
