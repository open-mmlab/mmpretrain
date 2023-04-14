# Copyright (c) OpenMMLab. All rights reserved.
import copy
import fnmatch
import os.path as osp
import warnings
from os import PathLike
from pathlib import Path
from typing import List, Union

from mmengine.config import Config
from modelindex.load_model_index import load
from modelindex.models.Model import Model


class ModelHub:
    """A hub to host the meta information of all pre-defined models."""
    _models_dict = {}
    __mmpretrain_registered = False

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
            metainfo.config = cls._expand_config_path(metainfo, config_prefix)
            cls._models_dict[model_name] = metainfo

    @classmethod
    def get(cls, model_name):
        """Get the model's metainfo by the model name.

        Args:
            model_name (str): The name of model.

        Returns:
            modelindex.models.Model: The metainfo of the specified model.
        """
        cls._register_mmpretrain_models()
        # lazy load config
        metainfo = copy.deepcopy(cls._models_dict.get(model_name.lower()))
        if metainfo is None:
            raise ValueError(
                f'Failed to find model "{model_name}". please use '
                '`mmpretrain.list_models` to get all available names.')
        if isinstance(metainfo.config, str):
            metainfo.config = Config.fromfile(metainfo.config)
        return metainfo

    @staticmethod
    def _expand_config_path(metainfo: Model,
                            config_prefix: Union[str, PathLike] = None):
        if config_prefix is None:
            config_prefix = osp.dirname(metainfo.filepath)

        if metainfo.config is None or osp.isabs(metainfo.config):
            config_path: str = metainfo.config
        else:
            config_path = osp.abspath(osp.join(config_prefix, metainfo.config))

        return config_path

    @classmethod
    def _register_mmpretrain_models(cls):
        # register models in mmpretrain
        if not cls.__mmpretrain_registered:
            from importlib_metadata import distribution
            root = distribution('mmpretrain').locate_file('mmpretrain')
            model_index_path = root / '.mim' / 'model-index.yml'
            ModelHub.register_model_index(
                model_index_path, config_prefix=root / '.mim')
            cls.__mmpretrain_registered = True

    @classmethod
    def has(cls, model_name):
        """Whether a model name is in the ModelHub."""
        return model_name in cls._models_dict


def get_model(model: Union[str, Config],
              pretrained: Union[str, bool] = False,
              device=None,
              **kwargs):
    """Get a pre-defined model or create a model from config.

    Args:
        model (str | Config): The name of model, the config file path or a
            config instance.
        pretrained (bool | str): When use name to specify model, you can
            use ``True`` to load the pre-defined pretrained weights. And you
            can also use a string to specify the path or link of weights to
            load. Defaults to False.
        device (str | torch.device | None): Transfer the model to the target
            device. Defaults to None.
        **kwargs: Other keyword arguments of the model config.

    Returns:
        mmengine.model.BaseModel: The result model.

    Examples:
        Get a ResNet-50 model and extract images feature:

        >>> import torch
        >>> from mmpretrain import get_model
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

        >>> from mmpretrain import get_model, inference_model
        >>> model = get_model('swin-base_16xb64_in1k', pretrained=True)
        >>> result = inference_model(model, 'demo/demo.JPEG')
        >>> print(result['pred_class'])
        'sea snake'
    """  # noqa: E501
    if isinstance(model, Config):
        config = copy.deepcopy(model)
        if pretrained is True and 'load_from' in config:
            pretrained = config.load_from
    elif isinstance(model, (str, PathLike)) and Path(model).suffix:
        config = Config.fromfile(model)
        if pretrained is True and 'load_from' in config:
            pretrained = config.load_from
    elif isinstance(model, str):
        metainfo = ModelHub.get(model)
        config = metainfo.config
        if pretrained is True and metainfo.weights is not None:
            pretrained = metainfo.weights
    else:
        raise TypeError('model must be a name, a path or a Config object, '
                        f'but got {type(config)}')

    if pretrained is True:
        warnings.warn('Unable to find pre-defined checkpoint of the model.')
        pretrained = None
    elif pretrained is False:
        pretrained = None

    if kwargs:
        config.merge_from_dict({'model': kwargs})
    config.model.setdefault('data_preprocessor',
                            config.get('data_preprocessor', None))

    from mmpretrain.registry import MODELS
    model = MODELS.build(config.model)

    dataset_meta = {}
    if pretrained:
        # Mapping the weights to GPU may cause unexpected video memory leak
        # which refers to https://github.com/open-mmlab/mmdetection/pull/6405
        from mmengine.runner import load_checkpoint
        checkpoint = load_checkpoint(model, pretrained, map_location='cpu')
        if 'dataset_meta' in checkpoint.get('meta', {}):
            # mmpretrain 1.x
            dataset_meta = checkpoint['meta']['dataset_meta']
        elif 'CLASSES' in checkpoint.get('meta', {}):
            # mmcls 0.x
            dataset_meta = {'classes': checkpoint['meta']['CLASSES']}

    if len(dataset_meta) == 0 and 'test_dataloader' in config:
        from mmpretrain.registry import DATASETS
        dataset_class = DATASETS.get(config.test_dataloader.dataset.type)
        dataset_meta = getattr(dataset_class, 'METAINFO', {})

    model.dataset_meta = dataset_meta
    model.config = config  # save the config in the model for convenience
    model.to(device)
    model.eval()
    return model


def init_model(config, checkpoint=None, device=None, **kwargs):
    """Initialize a classifier from config file (deprecated).

    It's only for compatibility, please use :func:`get_model` instead.

    Args:
        config (str | :obj:`mmengine.Config`): Config file path or the config
            object.
        checkpoint (str, optional): Checkpoint path. If left as None, the model
            will not load any weights.
        device (str | torch.device | None): Transfer the model to the target
            device. Defaults to None.
        **kwargs: Other keyword arguments of the model config.

    Returns:
        nn.Module: The constructed model.
    """
    return get_model(config, checkpoint, device, **kwargs)


def list_models(pattern=None, exclude_patterns=None, task=None) -> List[str]:
    """List all models available in MMPretrain.

    Args:
        pattern (str | None): A wildcard pattern to match model names.
            Defaults to None.
        exclude_patterns (list | None): A list of wildcard patterns to
            exclude names from the matched names. Defaults to None.
        task (str | none): The evaluation task of the model.

    Returns:
        List[str]: a list of model names.

    Examples:
        List all models:

        >>> from mmpretrain import list_models
        >>> list_models()

        List ResNet-50 models on ImageNet-1k dataset:

        >>> from mmpretrain import list_models
        >>> list_models('resnet*in1k')
        ['resnet50_8xb32_in1k',
         'resnet50_8xb32-fp16_in1k',
         'resnet50_8xb256-rsb-a1-600e_in1k',
         'resnet50_8xb256-rsb-a2-300e_in1k',
         'resnet50_8xb256-rsb-a3-100e_in1k']

        List Swin-Transformer models trained from stratch and exclude
        Swin-Transformer-V2 models:

        >>> from mmpretrain import list_models
        >>> list_models('swin', exclude_patterns=['swinv2', '*-pre'])
        ['swin-base_16xb64_in1k',
         'swin-base_3rdparty_in1k',
         'swin-base_3rdparty_in1k-384',
         'swin-large_8xb8_cub-384px',
         'swin-small_16xb64_in1k',
         'swin-small_3rdparty_in1k',
         'swin-tiny_16xb64_in1k',
         'swin-tiny_3rdparty_in1k']

        List all EVA models for image classification task.

        >>> from mmpretrain import list_models
        >>> list_models('eva', task='Image Classification')
        ['eva-g-p14_30m-in21k-pre_3rdparty_in1k-336px',
         'eva-g-p14_30m-in21k-pre_3rdparty_in1k-560px',
         'eva-l-p14_mim-in21k-pre_3rdparty_in1k-196px',
         'eva-l-p14_mim-in21k-pre_3rdparty_in1k-336px',
         'eva-l-p14_mim-pre_3rdparty_in1k-196px',
         'eva-l-p14_mim-pre_3rdparty_in1k-336px']
    """
    ModelHub._register_mmpretrain_models()
    matches = set(ModelHub._models_dict.keys())

    if pattern is not None:
        # Always match keys with any postfix.
        matches = set(fnmatch.filter(matches, pattern + '*'))

    exclude_patterns = exclude_patterns or []
    for exclude_pattern in exclude_patterns:
        exclude = set(fnmatch.filter(matches, exclude_pattern + '*'))
        matches = matches - exclude

    if task is not None:
        task_matches = []
        for key in matches:
            metainfo = ModelHub._models_dict[key]
            if metainfo.results is None and task == 'null':
                task_matches.append(key)
            elif task in [result.task for result in metainfo.results]:
                task_matches.append(key)
        matches = task_matches

    return sorted(list(matches))


def inference_model(model, input_, **kwargs):
    """Inference an image with the inferencer.

    Automatically select inferencer to inference according to the type of
    model. It's a shortcut for a quick start, and for advanced usage, please
    use the correspondding inferencer class.

    Here is the mapping from model type to inferencer:

    - :class:`~mmpretrain.models.ImageClassifier`: :class:`ImageClassificationInferencer`.
    - :class:`~mmpretrain.models.ImageToImageRetriever`: :class:`ImageToImageRetrievalInferencer`.

    Args:
        model (BaseModel | str | Config): The loaded model, the model
            name or the config of the model.
        input_ (str | ndarray): The image path or loaded image.
        **kwargs: Other keyword arguments to initialize the correspondding
            inferencer.

    Returns:
        result (dict): The inference results.
    """  # noqa: E501
    from mmengine.model import BaseModel

    if not isinstance(model, BaseModel):
        model = get_model(model, pretrained=True)

    import mmpretrain.models
    from .image_classification import ImageClassificationInferencer
    from .image_retrieval import ImageRetrievalInferencer

    if isinstance(model, mmpretrain.models.ImageClassifier):
        inferencer = ImageClassificationInferencer(model, **kwargs)
    elif isinstance(model, mmpretrain.models.ImageToImageRetriever):
        inferencer = ImageRetrievalInferencer(model, **kwargs)
    else:
        raise NotImplementedError(f'No available inferencer for {type(model)}')
    return inferencer(input_)[0]
