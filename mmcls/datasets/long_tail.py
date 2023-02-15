# Copyright (c) OpenMMLab. All rights reserved.
import math

import numpy as np

from mmcls.registry import DATASETS
from .cifar import CIFAR10, CIFAR100
from .imagenet import ImageNet


class ImbalancedDatasetMixin:
    """A mixin class for Imbalance Dataset.

    Args:
        imb_ratio (int): imbalance ratio, representing the ratio between
            the sample size of the most sampled class and the sample size of
            the least sampled class. Defaults to 10.
        imb_type (str): imbalance type, choose from 'exp' and 'step'.
            Defaults to 'exp'.
    """

    def __init__(self,
                 *args,
                 imb_ratio: int = 10,
                 imb_type: str = 'exp',
                 **kwargs):
        assert imb_ratio > 0 and isinstance(imb_ratio, int)
        assert imb_type in ('exp', 'step')
        self.imb_ratio = 1 / imb_ratio
        self.imb_type = imb_type

        super().__init__(*args, **kwargs)

    def load_data_list(self):
        """Load images and ground truth labels."""
        data_list = super().load_data_list()
        # only unbalanced sampling of the training dataset
        if not self.test_mode:
            data_list = self._gen_imbalanced_data_list(data_list)
        return data_list

    def _get_class_dict(self, data_list):
        class_dict = dict()
        for i, anno in enumerate(data_list):
            cat_id = anno['gt_label']
            if cat_id not in class_dict:
                class_dict[cat_id] = []
            class_dict[cat_id].append(i)
        sorted_class_dict = sorted(
            class_dict.items(), key=lambda x: len(x[1]), reverse=True)
        return sorted_class_dict

    def _get_sample_num_per_cls(self, class_sampeIds_list, imb_type,
                                imb_factor):
        cls_idx_list = [item[0] for item in class_sampeIds_list]
        cls_num = len(cls_idx_list)
        sample_num_list = [len(item[1]) for item in class_sampeIds_list]
        sample_num_max = sample_num_list[0]
        sample_num_each_cls = []
        if imb_type == 'exp':
            for i, cls_idx in enumerate(cls_idx_list):
                num = sample_num_max * (imb_factor**(i / (cls_num - 1.0)))
                sample_num_each_cls.append(math.ceil(num))
        elif imb_type == 'step':
            for i, cls_idx in enumerate(cls_idx_list):
                # in 'step' mode, sample_num_each_cls is an equivariate series
                k = (1 - imb_factor) / (cls_num - 1.0)
                num = math.ceil(sample_num_max * (1 - k * i))
                sample_num_each_cls.append(num)
        return sample_num_each_cls

    def _gen_imbalanced_data_list(self, data_list):
        """generate new imbalanced data_list from data_list."""
        class_sampeIds_list = self._get_class_dict(data_list)
        img_num_per_cls = self._get_sample_num_per_cls(class_sampeIds_list,
                                                       self.imb_type,
                                                       self.imb_ratio)
        new_data_list = []
        for i, (the_class, the_sampeIds) in enumerate(class_sampeIds_list):
            the_sample_num = img_num_per_cls[i]
            np.random.seed(the_class)
            np.random.shuffle(the_sampeIds)
            reserved_samples = the_sampeIds[:the_sample_num]
            for idx in reserved_samples:
                new_data_list.append(data_list[idx])

        return new_data_list


@DATASETS.register_module()
class LongTailCIFAR100(ImbalancedDatasetMixin, CIFAR100):
    """`Long Tail CIFAR100 <https://www.cs.toronto.edu/~kriz/cifar.html>`_
    Dataset.

    Args:
        data_prefix (str): Prefix for data.
        test_mode (bool): ``test_mode=True`` means in test phase.
            It determines to use the training set or test set.
        metainfo (dict, optional): Meta information for dataset, such as
            categories information. Defaults to None.
        data_root (str): The root directory for ``data_prefix``.
            Defaults to ''.
        download (bool): Whether to download the dataset if not exists.
            Defaults to True.
        imb_ratio (int): imbalance ratio, representing the ratio between
            the sample size of the most sampled class and the sample size of
            the least sampled class. Defaults to 10.
        imb_type (str): imbalance type, choose from 'exp' and 'step'.
            Defaults to 'exp'.
        **kwargs: Other keyword arguments in :class:`BaseDataset`.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


@DATASETS.register_module()
class LongTailCIFAR10(ImbalancedDatasetMixin, CIFAR10):
    """`Long Tail CIFAR10 <https://www.cs.toronto.edu/~kriz/cifar.html>`_
    Dataset.

    Args:
        data_prefix (str): Prefix for data.
        test_mode (bool): ``test_mode=True`` means in test phase.
            It determines to use the training set or test set.
        metainfo (dict, optional): Meta information for dataset, such as
            categories information. Defaults to None.
        data_root (str): The root directory for ``data_prefix``.
            Defaults to ''.
        download (bool): Whether to download the dataset if not exists.
            Defaults to True.
        imb_ratio (int): imbalance ratio, representing the ratio between
            the sample size of the most sampled class and the sample size of
            the least sampled class. Defaults to 10.
        imb_type (str): imbalance type, choose from 'exp' and 'step'.
            Defaults to 'exp'.
        **kwargs: Other keyword arguments in :class:`BaseDataset`.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


@DATASETS.register_module()
class LongTailImageNet(ImbalancedDatasetMixin, ImageNet):
    """`Long Tail ImageNet <http://www.image-net.org>`_ Dataset.

    Args:
        ann_file (str): Annotation file path. Defaults to ''.
        metainfo (dict, optional): Meta information for dataset, such as class
            information. Defaults to None.
        data_root (str): The root directory for ``data_prefix`` and
            ``ann_file``. Defaults to ''.
        data_prefix (str | dict): Prefix for training data. Defaults to ''.
        imb_ratio (int): imbalance ratio, representing the ratio between
            the sample size of the most sampled class and the sample size of
            the least sampled class. Defaults to 10.
        imb_type (str): imbalance type, choose from 'exp' and 'step'.
            Defaults to 'exp'.
        **kwargs: Other keyword arguments in :class:`CustomDataset` and
            :class:`BaseDataset`.
    """  # noqa: E501

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
