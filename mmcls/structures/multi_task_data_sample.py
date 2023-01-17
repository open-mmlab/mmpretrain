# Copyright (c) OpenMMLab. All rights reserved.

from mmengine.structures import BaseDataElement


class MultiTaskDataSample(BaseDataElement):

    @property
    def tasks(self):
        return self._data_fields
