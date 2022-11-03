# Copyright (c) OpenMMLab. All rights reserved.
from typing import List

from mmengine.model import BaseTTAModel
from mmengine.structures import BaseDataElement

from mmcls.registry import MODELS
from mmcls.structures import ClsDataSample


@MODELS.register_module()
class AverageScoreTTAModel(BaseTTAModel):

    def merge_preds(
        self,
        data_samples_list: List[List[ClsDataSample]],
    ) -> List[BaseDataElement]:
        merged_data_samples = []
        for data_samples in data_samples_list:
            merged_data_samples.append(self.merge_single_sample(data_samples))
        return merged_data_samples

    def merge_single_sample(self, data_samples):
        merged_data_sample: ClsDataSample = data_samples[0].new()
        merged_score = sum(data_sample.pred_label.score
                           for data_sample in data_samples) / len(data_samples)
        merged_data_sample.set_pred_score(merged_score)
        return merged_data_sample
