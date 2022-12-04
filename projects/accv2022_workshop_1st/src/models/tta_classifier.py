# Copyright (c) OpenMMLab. All rights reserved.
from typing import Union

from mmcls.models import ImageClassifier
from mmcls.registry import MODELS


@MODELS.register_module(force=True)
class TTAImageClassifier(ImageClassifier):

    def __init__(self, tta=True, **kwargs):
        super().__init__(**kwargs)
        self.tta = tta

    def val_step(self, data: Union[tuple, dict, list]) -> list:
        """Gets the predictions of given data.

        Calls ``self.data_preprocessor(data, False)`` and
        ``self(inputs, data_sample, mode='predict')`` in order. Return the
        predictions which will be passed to evaluator.

        Args:
            data (dict or tuple or list): Data sampled from dataset.

        Returns:
            list: The predictions of given data.
        """
        data = self.data_preprocessor(data, False)
        return self._run_forward(data, mode='predict')

    def test_step(self, data: Union[dict, tuple, list]) -> list:
        """``BaseModel`` implements ``test_step`` the same as ``val_step``.

        Args:
            data (dict or tuple or list): Data sampled from dataset.

        Returns:
            list: The predictions of given data.
        """
        data = self.data_preprocessor(data, False)
        if self.tta:
            inputs, data_samples = data['inputs'], data['data_samples']
            ori_out = self(inputs)
            flip_out = self(inputs.flip((3, )))
            out = (ori_out + flip_out) / 2
            return self.head._get_predictions(out, data_samples)
        else:
            return self._run_forward(data, mode='predict')
