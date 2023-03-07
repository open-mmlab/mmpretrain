# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional
from urllib.parse import urljoin

import mmengine.dist as dist
import numpy as np
from mmengine.fileio import LocalBackend, exists, get_file_backend, join_path
from scipy.io import matlab

from mmcls.registry import DATASETS
from .base_dataset import BaseDataset
from .categories import MNIST_CATEGORITES
from .utils import download_url


@DATASETS.register_module()
class SVHN(BaseDataset):
    """`SVHN <http://ufldl.stanford.edu/housenumbers>`_ Dataset.
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
        **kwargs: Other keyword arguments in :class:`BaseDataset`.
    """  # noqa: E501

    url_prefix = 'http://ufldl.stanford.edu/housenumbers'
    # train images and labels
    train_list = [
        ['train_32x32.mat', 'e26dedcc434d2e4c54c9b2d4a06d8373'],
    ]
    # test images and labels
    test_list = [
        ['test_32x32.mat', 'eb5a983be6a315427106f1b164d9cef3'],
    ]
    # extra images and labels, but it is not used
    extra_list = [
        ['extra_32x32.mat', 'a93ce644f1a588dc4d68dda5feec44a7'],
    ]
    METAINFO = {'classes': MNIST_CATEGORITES}

    def __init__(self,
                 data_prefix: str,
                 test_mode: bool,
                 metainfo: Optional[dict] = None,
                 data_root: str = '',
                 download: bool = True,
                 **kwargs):
        self.download = download
        super().__init__(
            # The SVHN dataset doesn't need specify annotation file
            ann_file='',
            metainfo=metainfo,
            data_root=data_root,
            data_prefix=dict(root=data_prefix),
            test_mode=test_mode,
            **kwargs)

    def load_data_list(self):
        """Load images and ground truth labels."""
        root = self.data_prefix['root']
        backend = get_file_backend(root, enable_singleton=True)

        if dist.is_main_process() and not self._check_exists():
            if not isinstance(backend, LocalBackend):
                raise RuntimeError(f'The dataset on {root} is not integrated, '
                                   f'please manually handle it.')

            if self.download:
                self._download()
            else:
                raise RuntimeError(
                    f'Cannot find {self.__class__.__name__} dataset in '
                    f"{self.data_prefix['root']}, you can specify "
                    '`download=True` to download automatically.')

        dist.barrier()
        assert self._check_exists(), \
            'Download failed or shared storage is unavailable. Please ' \
            f'download the dataset manually through {self.url_prefix}.'

        if not self.test_mode:
            file_list = self.train_list
        else:
            file_list = self.test_list

        # load data from mat file
        mat = matlab.loadmat(join_path(root, file_list[0][0]))
        imgs = np.transpose(mat['X'], (3, 0, 1, 2))  # convert HWCN to NHWC
        gt_labels = np.squeeze(mat['y'])  # convert N1 to N
        gt_labels[gt_labels == 10] = 0  # overwrite label 10 to 0

        data_list = list()
        for img, gt_label in zip(imgs, gt_labels):
            info = dict(img=img, gt_label=gt_label)
            data_list.append(info)
        return data_list

    def _check_exists(self):
        """Check the exists of data files."""
        root = self.data_prefix['root']

        for filename, _ in (self.train_list + self.test_list):
            # get extracted filename of data
            fpath = join_path(root, filename)
            if not exists(fpath):
                return False
        return True

    def _download(self):
        """Download and extract data files."""
        root = self.data_prefix['root']

        for filename, md5 in (self.train_list + self.test_list):
            url = urljoin(self.url_prefix, filename)
            download_url(url, root, filename=filename, md5=md5)
