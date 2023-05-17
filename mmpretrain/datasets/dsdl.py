# Copyright (c) OpenMMLab. All rights reserved.
import os
from typing import List

from mmpretrain.registry import DATASETS
from .base_dataset import BaseDataset

try:
    from dsdl.dataset import DSDLDataset
except ImportError:
    DSDLDataset = None


@DATASETS.register_module()
class DSDLClsDataset(BaseDataset):
    """Dataset for dsdl classification.

    Args:
        specific_key_path(dict): Path of specific key which can not
            be loaded by it's field name.
        pre_transform(dict): pre-transform functions before loading.
    """

    METAINFO = {}

    def __init__(self,
                 specific_key_path: dict = {},
                 pre_transform: dict = {},
                 **kwargs) -> None:

        if DSDLDataset is None:
            raise RuntimeError(
                'Package dsdl is not installed. Please run "pip install dsdl".'
            )

        loc_config = dict(type='LocalFileReader', working_dir='')
        if kwargs.get('data_root'):
            kwargs['ann_file'] = os.path.join(kwargs['data_root'],
                                              kwargs['ann_file'])
        self.required_fields = ['Image', 'Label']

        self.dsdldataset = DSDLDataset(
            dsdl_yaml=kwargs['ann_file'],
            location_config=loc_config,
            required_fields=self.required_fields,
            specific_key_path=specific_key_path,
            transform=pre_transform,
        )

        BaseDataset.__init__(self, **kwargs)

    def load_data_list(self) -> List[dict]:
        """Load data info from a dsdl yaml file named as ``self.ann_file``
        Returns:
            List[dict]: A list of data list.
        """
        self._metainfo['classes'] = tuple(self.dsdldataset.class_names)

        data_list = []

        for i, data in enumerate(self.dsdldataset):
            if len(data['Label']) == 1:
                label_index = data['Label'][0].index_in_domain() - 1
            else:
                # multi labels
                label_index = [
                    category.index_in_domain() - 1
                    for category in data['Label']
                ]
            datainfo = dict(
                img_path=os.path.join(self.data_prefix['img_path'],
                                      data['Image'][0].location),
                gt_label=label_index)
            data_list.append(datainfo)
        return data_list
