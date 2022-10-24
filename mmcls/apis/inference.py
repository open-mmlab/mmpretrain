# Copyright (c) OpenMMLab. All rights reserved.
import warnings

import torch
from mmengine.config import Config
from mmengine.dataset import Compose, pseudo_collate
from mmengine.runner import load_checkpoint

from mmcls.models import build_classifier


def init_model(config,
               checkpoint=None,
               device='cuda:0',
               classes=None,
               options=None):
    """Initialize a classifier from config file.

    Args:
        config (str or :obj:`mmengine.Config`): Config file path or the config
            object.
        checkpoint (str, optional): Checkpoint path. If left as None, the model
            will not load any weights.
        classes (List[str], optional): Class names. If left as None, the model
            will get classes from checkpoint or set default ImageNet1k classes.
        options (dict): Options to override some settings in the used config.

    Returns:
        nn.Module: The constructed classifier.
    """
    if isinstance(config, str):
        config = Config.fromfile(config)
    elif not isinstance(config, Config):
        raise TypeError('config must be a filename or Config object, '
                        f'but got {type(config)}')
    if options is not None:
        config.merge_from_dict(options)
    config.model.setdefault('data_preprocessor',
                            config.get('data_preprocessor', None))
    model = build_classifier(config.model)
    if checkpoint is not None:
        # Mapping the weights to GPU may cause unexpected video memory leak
        # which refers to https://github.com/open-mmlab/mmdetection/pull/6405
        checkpoint = load_checkpoint(model, checkpoint, map_location='cpu')
    _set_model_classes(model, classes, checkpoint)  # set attr model.CLASSES
    model.cfg = config  # save the config in the model for convenience
    model.to(device)
    model.eval()
    return model


def inference_model(model, img):
    """Inference image(s) with the classifier.

    Args:
        model (BaseClassifier): The loaded classifier.
        img (str/ndarray): The image filename or loaded image.

    Returns:
        result (dict): The classification results that contains
            `class_name`, `pred_label` and `pred_score`.
    """
    cfg = model.cfg
    # build the data pipeline
    test_pipeline_cfg = cfg.test_dataloader.dataset.pipeline
    if isinstance(img, str):
        if test_pipeline_cfg[0]['type'] != 'LoadImageFromFile':
            test_pipeline_cfg.insert(0, dict(type='LoadImageFromFile'))
        data = dict(img_path=img)
    else:
        if test_pipeline_cfg[0]['type'] == 'LoadImageFromFile':
            test_pipeline_cfg.pop(0)
        data = dict(img=img)
    test_pipeline = Compose(test_pipeline_cfg)
    data = test_pipeline(data)
    data = pseudo_collate([data])

    # forward the model
    with torch.no_grad():
        prediction = model.val_step(data)[0].pred_label
        pred_scores = prediction.score.tolist()
        pred_score = torch.max(prediction.score).item()
        pred_label = prediction.label.item()
        result = {
            'pred_label': pred_label,
            'pred_score': float(pred_score),
            'pred_scores': pred_scores
        }
    if hasattr(model, 'CLASSES') and model.CLASSES is not None:
        result['pred_class'] = model.CLASSES[result['pred_label']]
    return result


def _set_model_classes(model, classes, checkpoint):
    """set."""
    if classes is not None:
        # set to classes if classes is set.
        model.CLASSES = classes
    elif checkpoint is not None:
        # load CLASSES from checkpoint
        meta_info = checkpoint.get('meta', {})
        if 'dataset_meta' in meta_info and 'classes' in meta_info[
                'dataset_meta']:
            # mmcls 1.x
            model.CLASSES = meta_info['dataset_meta']['classes']
        elif 'CLASSES' in meta_info:
            # mmcls < 1.x
            model.CLASSES = meta_info['CLASSES']
    else:
        # set to default ImageNet-1k classes names
        from mmcls.datasets.categories import IMAGENET_CATEGORIES
        warnings.simplefilter('once')
        warnings.warn('Class names are not saved in the checkpoint\'s '
                      'meta data, use imagenet by default.')
        model.CLASSES = IMAGENET_CATEGORIES
