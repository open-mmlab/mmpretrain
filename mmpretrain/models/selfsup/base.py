# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Optional, Union

import torch
from mmengine.model import BaseModel
from torch import nn

from mmpretrain.registry import MODELS
from mmpretrain.structures import DataSample


class BaseSelfSupervisor(BaseModel):
    """BaseModel for Self-Supervised Learning.

    All algorithms should inherit this module.

    Args:
        backbone (dict): The backbone module. See
            :mod:`mmcls.models.backbones`.
        neck (dict, optional): The neck module to process features from
            backbone. See :mod:`mmcls.models.necks`. Defaults to None.
        head (dict, optional): The head module to do prediction and calculate
            loss from processed features. See :mod:`mmcls.models.heads`.
            Notice that if the head is not set, almost all methods cannot be
            used except :meth:`extract_feat`. Defaults to None.
        target_generator: (dict, optional): The target_generator module to
            generate targets for self-supervised learning optimization, such as
            HOG, extracted features from other modules(DALL-E, CLIP), etc.
        pretrained (str, optional): The pretrained checkpoint path, support
            local path and remote path. Defaults to None.
        data_preprocessor (Union[dict, nn.Module], optional): The config for
            preprocessing input data. If None or no specified type, it will use
            "SelfSupDataPreprocessor" as type.
            See :class:`SelfSupDataPreprocessor` for more details.
            Defaults to None.
        init_cfg (dict, optional): the config to control the initialization.
            Defaults to None.
    """

    def __init__(self,
                 backbone: dict,
                 neck: Optional[dict] = None,
                 head: Optional[dict] = None,
                 target_generator: Optional[dict] = None,
                 pretrained: Optional[str] = None,
                 data_preprocessor: Optional[Union[dict, nn.Module]] = None,
                 init_cfg: Optional[dict] = None):

        if pretrained is not None:
            init_cfg = dict(type='Pretrained', checkpoint=pretrained)

        if data_preprocessor is None:
            data_preprocessor = {}
        # The build process is in MMEngine, so we need to add scope here.
        data_preprocessor.setdefault('type',
                                     'mmpretrain.SelfSupDataPreprocessor')

        super().__init__(
            init_cfg=init_cfg, data_preprocessor=data_preprocessor)

        self.backbone = MODELS.build(backbone)

        if neck is not None:
            self.neck = MODELS.build(neck)

        if head is not None:
            self.head = MODELS.build(head)

        if target_generator is not None:
            self.target_generator = MODELS.build(target_generator)

    @property
    def with_neck(self) -> bool:
        """Check if the model has a neck module."""
        return hasattr(self, 'neck') and self.neck is not None

    @property
    def with_head(self) -> bool:
        """Check if the model has a head module."""
        return hasattr(self, 'head') and self.head is not None

    @property
    def with_target_generator(self) -> bool:
        """Check if the model has a target_generator module."""
        return hasattr(
            self, 'target_generator') and self.target_generator is not None

    def forward(self,
                inputs: torch.Tensor,
                data_samples: Optional[List[DataSample]] = None,
                mode: str = 'tensor'):
        """Returns losses or predictions of training, validation, testing, and
        simple inference process.

        This module overwrites the abstract method in ``BaseModel``.

        Args:
            inputs (torch.Tensor): batch input tensor collated by
                :attr:`data_preprocessor`.
            data_samples (List[DataSample], optional):
                data samples collated by :attr:`data_preprocessor`.
            mode (str): mode should be one of ``loss``, ``predict`` and
                ``tensor``.

                - ``loss``: Called by ``train_step`` and return loss ``dict``
                  used for logging
                - ``predict``: Called by ``val_step`` and ``test_step``
                  and return list of ``DataSample`` results used for
                  computing metric.
                - ``tensor``: Called by custom use to get ``Tensor`` type
                  results.

        Returns:
            ForwardResults (dict or list):
              - If ``mode == loss``, return a ``dict`` of loss tensor used
                for backward and logging.
              - If ``mode == predict``, return a ``list`` of
                :obj:`DataSample` for computing metric
                and getting inference result.
              - If ``mode == tensor``, return a tensor or ``tuple`` of tensor
                or ``dict of tensor for custom use.
        """
        if mode == 'tensor':
            feats = self.extract_feat(inputs)
            return feats
        elif mode == 'loss':
            return self.loss(inputs, data_samples)
        elif mode == 'predict':
            return self.predict(inputs, data_samples)
        else:
            raise RuntimeError(f'Invalid mode "{mode}".')

    def extract_feat(self, inputs: torch.Tensor):
        """Extract features from the input tensor with shape (N, C, ...).

        This is a abstract method, and subclass should overwrite this methods
        if needed.

        Args:
            inputs (Tensor): A batch of inputs. The shape of it should be
                ``(num_samples, num_channels, *img_shape)``.

        Returns:
            tuple | Tensor: The output of specified stage.
            The output depends on detailed implementation.
        """
        raise NotImplementedError

    def loss(self, inputs: torch.Tensor,
             data_samples: List[DataSample]) -> dict:
        """Calculate losses from a batch of inputs and data samples.

        This is a abstract method, and subclass should overwrite this methods
        if needed.

        Args:
            inputs (torch.Tensor): The input tensor with shape
                (N, C, ...) in general.
            data_samples (List[DataSample]): The annotation data of
                every samples.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        raise NotImplementedError

    def predict(self,
                inputs: tuple,
                data_samples: Optional[List[DataSample]] = None,
                **kwargs) -> List[DataSample]:
        """Predict results from the extracted features.

        This module returns the logits before loss, which are used to compute
        all kinds of metrics. This is a abstract method, and subclass should
        overwrite this methods if needed.

        Args:
            feats (tuple): The features extracted from the backbone.
            data_samples (List[DataSample], optional): The annotation
                data of every samples. Defaults to None.
            **kwargs: Other keyword arguments accepted by the ``predict``
                method of :attr:`head`.
        """
        raise NotImplementedError
