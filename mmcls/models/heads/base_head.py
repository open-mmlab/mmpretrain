# Copyright (c) OpenMMLab. All rights reserved.
import warnings
from abc import ABCMeta, abstractmethod
from typing import List, Optional, Tuple

import torch
from mmengine.model import BaseModule
from mmengine.structures import BaseDataElement
from mmengine.utils import digit_version
from torch import nn


class BaseHead(BaseModule, metaclass=ABCMeta):
    """Base head.

    Args:
        init_cfg (dict, optional): The extra init config of layers.
            Defaults to None.
        skip_init_weights (bool): Whether to skip init_weights. When using Lazy
            modules, it should be skipped. Defaults to False.
    """

    def __init__(self,
                 init_cfg: Optional[dict] = None,
                 skip_init_weights: bool = False):
        super(BaseHead, self).__init__(init_cfg=init_cfg)
        self.skip_init_weights = skip_init_weights

    def init_weights(self):
        if not self.skip_init_weights:
            super().init_weights()

    def _create_linear(self, in_channels: Optional[int], num_classes: int):
        if in_channels is None:
            if digit_version(torch.__version__) >= digit_version('1.8.0'):
                warnings.warn(
                    'Head with in_channels is None, it uses LazyLinear '
                    'init_cfg is ignored and in_channels '
                    'is calculated automatically.')
                return nn.LazyLinear(num_classes)
            else:
                raise RuntimeError(
                    'torch.nn.LazyLinear is not available before 1.8.0')
        else:
            return nn.Linear(in_channels, num_classes)

    @abstractmethod
    def loss(self, feats: Tuple, data_samples: List[BaseDataElement]):
        """Calculate losses from the extracted features.

        Args:
            feats (tuple): The features extracted from the backbone.
            data_samples (List[BaseDataElement]): The annotation data of
                every samples.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        pass

    @abstractmethod
    def predict(self,
                feats: Tuple,
                data_samples: Optional[List[BaseDataElement]] = None):
        """Predict results from the extracted features.

        Args:
            feats (tuple): The features extracted from the backbone.
            data_samples (List[BaseDataElement], optional): The annotation
                data of every samples. If not None, set ``pred_label`` of
                the input data samples. Defaults to None.

        Returns:
            List[BaseDataElement]: A list of data samples which contains the
            predicted results.
        """
        pass
