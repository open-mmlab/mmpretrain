# Copyright (c) OpenMMLab. All rights reserved.
from typing import List

from mmengine.model import BaseTTAModel

from mmpretrain.registry import MODELS
from mmpretrain.structures import ClsDataSample


@MODELS.register_module()
class AverageClsScoreTTA(BaseTTAModel):

    def merge_preds(
        self,
        data_samples_list: List[List[ClsDataSample]],
    ) -> List[ClsDataSample]:
        """Merge predictions of enhanced data to one prediction.

        Args:
            data_samples_list (List[List[ClsDataSample]]): List of predictions
                of all enhanced data.

        Returns:
            List[ClsDataSample]: Merged prediction.
        """
        merged_data_samples = []
        for data_samples in data_samples_list:
            merged_data_samples.append(self._merge_single_sample(data_samples))
        return merged_data_samples

    def _merge_single_sample(self, data_samples):
        merged_data_sample: ClsDataSample = data_samples[0].new()
        merged_score = sum(data_sample.pred_label.score
                           for data_sample in data_samples) / len(data_samples)
        merged_data_sample.set_pred_score(merged_score)
        return merged_data_sample
