# Copyright (c) OpenMMLab. All right reserved.
import re
from collections import OrderedDict
from typing import List, Optional

import torch
import torch.nn.functional as F

from mmcls.registry import MODELS
from mmcls.structures import ClsDataSample
from .base import BaseClassifier


@MODELS.register_module()
class HuggingFaceClassifier(BaseClassifier):

    def __init__(self,
                 model_name,
                 pretrained=False,
                 *model_args,
                 loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
                 train_cfg: Optional[dict] = None,
                 data_preprocessor: Optional[dict] = None,
                 init_cfg: Optional[dict] = None,
                 **kwargs):
        if data_preprocessor is None:
            data_preprocessor = {}
        # The build process is in MMEngine, so we need to add scope here.
        data_preprocessor.setdefault('type', 'mmcls.ClsDataPreprocessor')

        if train_cfg is not None and 'augments' in train_cfg:
            # Set batch augmentations by `train_cfg`
            data_preprocessor['batch_augments'] = train_cfg

        super().__init__(
            init_cfg=init_cfg, data_preprocessor=data_preprocessor)

        from transformers import AutoConfig, AutoModelForImageClassification
        if pretrained:
            self.model = AutoModelForImageClassification.from_pretrained(
                model_name, *model_args, **kwargs)
        else:
            config = AutoConfig.from_pretrained(model_name, *model_args,
                                                **kwargs)
            self.model = AutoModelForImageClassification.from_config(config)
        self.loss_module = MODELS.build(loss)

        self._register_state_dict_hook(self._remove_state_dict_prefix)
        self._register_load_state_dict_pre_hook(self._add_state_dict_prefix)

    def forward(self, inputs, data_samples=None, mode='tensor'):
        if mode == 'tensor':
            return self.model(inputs)
        elif mode == 'loss':
            return self.loss(inputs, data_samples)
        elif mode == 'predict':
            return self.predict(inputs, data_samples)
        else:
            raise RuntimeError(f'Invalid mode "{mode}".')

    def extract_feat(self, inputs: torch.Tensor):
        raise NotImplementedError(
            "The HuggingFaceClassifier doesn't support extract feature yet.")

    def loss(self, inputs, data_samples, **kwargs):
        # The part can be traced by torch.fx
        cls_score = self.model(inputs).logits

        # The part can not be traced by torch.fx
        losses = self._get_loss(cls_score, data_samples, **kwargs)
        return losses

    def _get_loss(self, cls_score: torch.Tensor,
                  data_samples: List[ClsDataSample], **kwargs):
        """Unpack data samples and compute loss."""
        # Unpack data samples and pack targets
        if 'score' in data_samples[0].gt_label:
            # Batch augmentation may convert labels to one-hot format scores.
            target = torch.stack([i.gt_label.score for i in data_samples])
        else:
            target = torch.cat([i.gt_label.label for i in data_samples])

        # compute loss
        losses = dict()
        loss = self.loss_module(
            cls_score, target, avg_factor=cls_score.size(0), **kwargs)
        losses['loss'] = loss

        return losses

    def predict(self, inputs, data_samples):
        # The part can be traced by torch.fx
        cls_score = self(inputs).logits

        # The part can not be traced by torch.fx
        predictions = self._get_predictions(cls_score, data_samples)
        return predictions

    def _get_predictions(self, cls_score, data_samples):
        """Post-process the output of head.

        Including softmax and set ``pred_label`` of data samples.
        """
        pred_scores = F.softmax(cls_score, dim=1)
        pred_labels = pred_scores.argmax(dim=1, keepdim=True).detach()

        if data_samples is not None:
            for data_sample, score, label in zip(data_samples, pred_scores,
                                                 pred_labels):
                data_sample.set_pred_score(score).set_pred_label(label)
        else:
            data_samples = []
            for score, label in zip(pred_scores, pred_labels):
                data_samples.append(ClsDataSample().set_pred_score(
                    score).set_pred_label(label))

        return data_samples

    @staticmethod
    def _remove_state_dict_prefix(self, state_dict, prefix, local_metadata):
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            new_key = re.sub(f'^{prefix}model.', prefix, k)
            new_state_dict[new_key] = v
        return new_state_dict

    @staticmethod
    def _add_state_dict_prefix(state_dict, prefix, local_metadata, strict,
                               missing_keys, unexpected_keys, error_msgs):
        new_prefix = prefix + 'model.'
        for k in list(state_dict.keys()):
            new_key = re.sub(f'^{prefix}', new_prefix, k)
            state_dict[new_key] = state_dict[k]
            del state_dict[k]
