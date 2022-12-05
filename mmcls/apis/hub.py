# Copyright (c) OpenMMLab. All rights reserved.
import copy
import fnmatch
import os.path as osp
from os import PathLike
from pathlib import Path
from typing import List, Union

from mmengine.config import Config
from mmengine.utils import get_installed_path
from modelindex.load_model_index import load
from modelindex.models.Model import Model

from .inference import init_model


def _expand_config_path(metainfo: Model,
                        config_prefix: Union[str, PathLike] = None):
    if config_prefix is None:
        config_prefix = osp.dirname(metainfo.filepath)

    if metainfo.config is None or osp.isabs(metainfo.config):
        config_path: str = metainfo.config
    else:
        config_path = osp.abspath(osp.join(config_prefix, metainfo.config))

    return config_path


class ModelHub:
    """A hub to host the meta information of all pre-defined models."""
    _models_dict = {}

    @classmethod
    def register_model_index(cls,
                             model_index_path: Union[str, PathLike],
                             config_prefix: Union[str, PathLike, None] = None):
        """Parse the model-index file and register all models.

        Args:
            model_index_path (str | PathLike): The path of the model-index
                file.
            config_prefix (str | PathLike | None): The prefix of all config
                file paths in the model-index file.
        """
        model_index = load(str(model_index_path))
        model_index.build_models_with_collections()

        for metainfo in model_index.models:
            model_name = metainfo.name.lower()
            if metainfo.name in cls._models_dict:
                raise ValueError(
                    'The model name {} is conflict in {} and {}.'.format(
                        model_name, osp.abspath(metainfo.filepath),
                        osp.abspath(cls._models_dict[model_name].filepath)))
            metainfo.config = _expand_config_path(metainfo, config_prefix)
            cls._models_dict[model_name] = metainfo

    @classmethod
    def get(cls, model_name):
        """Get the model's metainfo by the model name.

        Args:
            model_name (str): The name of model.

        Returns:
            modelindex.models.Model: The metainfo of the specified model.
        """
        # lazy load config
        metainfo = copy.deepcopy(cls._models_dict.get(model_name.lower()))
        if metainfo is None:
            raise ValueError(f'Failed to find model {model_name}.')
        if isinstance(metainfo.config, str):
            metainfo.config = Config.fromfile(metainfo.config)
        return metainfo


# register models in mmcls
mmcls_root = Path(get_installed_path('mmcls'))
model_index_path = mmcls_root / '.mim' / 'model-index.yml'
ModelHub.register_model_index(
    model_index_path, config_prefix=mmcls_root / '.mim')


def get_model(model_name, pretrained=False, device=None, **kwargs):
    """Get a pre-defined model by the name of model.

    Args:
        model_name (str): The name of model.
        pretrained (bool | str): If True, load the pre-defined pretrained
            weights. If a string, load the weights from it. Defaults to False.
        device (str | torch.device | None): Transfer the model to the target
            device. Defaults to None.
        **kwargs: Other keyword arguments of the model config.

    Returns:
        mmengine.model.BaseModel: The result model.

    Examples:
        Get a ResNet-50 model and extract images feature:

        >>> import torch
        >>> from mmcls import get_model
        >>> inputs = torch.rand(16, 3, 224, 224)
        >>> model = get_model('resnet50_8xb32_in1k', pretrained=True, backbone=dict(out_indices=(0, 1, 2, 3)))
        >>> feats = model.extract_feat(inputs)
        >>> for feat in feats:
        ...     print(feat.shape)
        torch.Size([16, 256])
        torch.Size([16, 512])
        torch.Size([16, 1024])
        torch.Size([16, 2048])

        Get Swin-Transformer model with pre-trained weights and inference:

        >>> from mmcls import get_model, inference_model
        >>> model = get_model('swin-base_16xb64_in1k', pretrained=True)
        >>> result = inference_model(model, 'demo/demo.JPEG')
        >>> print(result['pred_class'])
        'sea snake'
    """  # noqa: E501
    metainfo = ModelHub.get(model_name)

    if isinstance(pretrained, str):
        ckpt = pretrained
    elif pretrained:
        if metainfo.weights is None:
            raise ValueError(
                f"The model {model_name} doesn't have pretrained weights.")
        ckpt = metainfo.weights
    else:
        ckpt = None

    if metainfo.config is None:
        raise ValueError(
            f"The model {model_name} doesn't support building by now.")
    model = init_model(
        metainfo.config, ckpt, device=device, options=dict(model=kwargs))
    return model


def list_models(pattern=None) -> List[str]:
    """List all models available in MMClassification.

    Args:
        pattern (str | None): A wildcard pattern to match model names.

    Returns:
        List[str]: a list of model names.

    Examples:
        List all models:

        >>> from mmcls import list_models
        >>> print(list_models())

        List ResNet-50 models on ImageNet-1k dataset:

        >>> from mmcls import list_models
        >>> print(list_models('resnet*in1k'))
        ['resnet50_8xb32_in1k',
         'resnet50_8xb32-fp16_in1k',
         'resnet50_8xb256-rsb-a1-600e_in1k',
         'resnet50_8xb256-rsb-a2-300e_in1k',
         'resnet50_8xb256-rsb-a3-100e_in1k']
    """
    if pattern is None:
        return sorted(list(ModelHub._models_dict.keys()))
    # Always match keys with any postfix.
    matches = fnmatch.filter(ModelHub._models_dict.keys(), pattern + '*')
    return matches
