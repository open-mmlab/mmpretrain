# Copyright (c) OpenMMLab. All rights reserved.
import gc
import pickle
import warnings
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np

from .builder import DATASETS
from .custom import CustomDataset


@DATASETS.register_module()
class ImageNet21k(CustomDataset):
    """ImageNet21k Dataset.

    Since the dataset ImageNet21k is extremely big, cantains 21k+ classes
    and 1.4B files. This class has improved the following points on the
    basis of the class ``ImageNet``, in order to save memory, we enable the
    ``serialize_data`` optional by default. With this option, the annotation
    won't be stored in the list ``data_infos``, but be serialized as an
    array.

    Args:
        data_prefix (str): The path of data directory.
        pipeline (Sequence[dict]): A list of dict, where each element
            represents a operation defined in :mod:`mmcls.datasets.pipelines`.
            Defaults to an empty tuple.
        classes (str | Sequence[str], optional): Specify names of classes.

            - If is string, it should be a file path, and the every line of
              the file is a name of a class.
            - If is a sequence of string, every item is a name of class.
            - If is None, the object won't have category information.
              (Not recommended)

            Defaults to None.
        ann_file (str, optional): The annotation file. If is string, read
            samples paths from the ann_file. If is None, find samples in
            ``data_prefix``. Defaults to None.
        serialize_data (bool): Whether to hold memory using serialized objects,
            when enabled, data loader workers can use shared RAM from master
            process instead of making a copy. Defaults to True.
        multi_label (bool): Not implement by now. Use multi label or not.
            Defaults to False.
        recursion_subdir(bool): Deprecated, and the dataset will recursively
            get all images now.
        test_mode (bool): In train mode or test mode. It's only a mark and
            won't be used in this class. Defaults to False.
        file_client_args (dict, optional): Arguments to instantiate a
            FileClient. See :class:`mmcv.fileio.FileClient` for details.
            If None, automatically inference from the specified path.
            Defaults to None.
    """

    IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif')
    CLASSES = None

    def __init__(self,
                 data_prefix: str,
                 pipeline: Sequence = (),
                 classes: Union[str, Sequence[str], None] = None,
                 ann_file: Optional[str] = None,
                 serialize_data: bool = True,
                 multi_label: bool = False,
                 recursion_subdir: bool = True,
                 test_mode=False,
                 file_client_args: Optional[dict] = None):
        assert recursion_subdir, 'The `recursion_subdir` option is ' \
            'deprecated. Now the dataset will recursively get all images.'
        if multi_label:
            raise NotImplementedError(
                'The `multi_label` option is not supported by now.')
        self.multi_label = multi_label
        self.serialize_data = serialize_data

        if ann_file is None:
            warnings.warn(
                'The ImageNet21k dataset is large, and scanning directory may '
                'consume long time. Considering to specify the `ann_file` to '
                'accelerate the initialization.', UserWarning)

        if classes is None:
            warnings.warn(
                'The CLASSES is not stored in the `ImageNet21k` class. '
                'Considering to specify the `classes` argument if you need '
                'do inference on the ImageNet-21k dataset', UserWarning)

        super().__init__(
            data_prefix=data_prefix,
            pipeline=pipeline,
            classes=classes,
            ann_file=ann_file,
            extensions=self.IMG_EXTENSIONS,
            test_mode=test_mode,
            file_client_args=file_client_args)

        if self.serialize_data:
            self.data_infos_bytes, self.data_address = self._serialize_data()
            # Empty cache for preventing making multiple copies of
            # `self.data_infos` when loading data multi-processes.
            self.data_infos.clear()
            gc.collect()

    def get_cat_ids(self, idx: int) -> List[int]:
        """Get category id by index.

        Args:
            idx (int): Index of data.

        Returns:
            cat_ids (List[int]): Image category of specified index.
        """

        return [int(self.get_data_info(idx)['gt_label'])]

    def get_data_info(self, idx: int) -> dict:
        """Get annotation by index.

        Args:
            idx (int): The index of data.

        Returns:
            dict: The idx-th annotation of the dataset.
        """
        if self.serialize_data:
            start_addr = 0 if idx == 0 else self.data_address[idx - 1].item()
            end_addr = self.data_address[idx].item()
            bytes = memoryview(self.data_infos_bytes[start_addr:end_addr])
            data_info = pickle.loads(bytes)
        else:
            data_info = self.data_infos[idx]

        return data_info

    def prepare_data(self, idx):
        data_info = self.get_data_info(idx)
        return self.pipeline(data_info)

    def _serialize_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Serialize ``self.data_infos`` to save memory when launching multiple
        workers in data loading. This function will be called in ``full_init``.

        Hold memory using serialized objects, and data loader workers can use
        shared RAM from master process instead of making a copy.

        Returns:
            Tuple[np.ndarray, np.ndarray]: serialize result and corresponding
            address.
        """

        def _serialize(data):
            buffer = pickle.dumps(data, protocol=4)
            return np.frombuffer(buffer, dtype=np.uint8)

        serialized_data_infos_list = [_serialize(x) for x in self.data_infos]
        address_list = np.asarray([len(x) for x in serialized_data_infos_list],
                                  dtype=np.int64)
        data_address: np.ndarray = np.cumsum(address_list)
        serialized_data_infos = np.concatenate(serialized_data_infos_list)

        return serialized_data_infos, data_address

    def __len__(self) -> int:
        """Get the length of filtered dataset and automatically call
        ``full_init`` if the  dataset has not been fully init.

        Returns:
            int: The length of filtered dataset.
        """
        if self.serialize_data:
            return len(self.data_address)
        else:
            return len(self.data_infos)
