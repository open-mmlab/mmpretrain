# Copyright (c) OpenMMLab. All rights reserved.
import warnings

import mmcv
import numpy as np
import torch
from mmcv.parallel import collate, scatter
from mmcv.runner import load_checkpoint

from mmcls.datasets.pipelines import Compose
from mmcls.models import build_classifier


def init_model(config, checkpoint=None, device='cuda:0', options=None):
    """Initialize a classifier from config file.

    Args:
        config (str or :obj:`mmcv.Config`): Config file path or the config
            object.
        checkpoint (str, optional): Checkpoint path. If left as None, the model
            will not load any weights.
        options (dict): Options to override some settings in the used config.

    Returns:
        nn.Module: The constructed classifier.
    """
    if isinstance(config, str):
        config = mmcv.Config.fromfile(config)
    elif not isinstance(config, mmcv.Config):
        raise TypeError('config must be a filename or Config object, '
                        f'but got {type(config)}')
    if options is not None:
        config.merge_from_dict(options)
    config.model.pretrained = None
    model = build_classifier(config.model)
    if checkpoint is not None:
        # Mapping the weights to GPU may cause unexpected video memory leak
        # which refers to https://github.com/open-mmlab/mmdetection/pull/6405
        checkpoint = load_checkpoint(model, checkpoint, map_location='cpu')
        if 'CLASSES' in checkpoint.get('meta', {}):
            model.CLASSES = checkpoint['meta']['CLASSES']
        else:
            from mmcls.datasets import ImageNet
            warnings.simplefilter('once')
            warnings.warn('Class names are not saved in the checkpoint\'s '
                          'meta data, use imagenet by default.')
            model.CLASSES = ImageNet.CLASSES
    model.cfg = config  # save the config in the model for convenience
    model.to(device)
    model.eval()
    return model


def inference_model(model, img):
    """Inference image(s) with the classifier.

    Args:
        model (nn.Module): The loaded classifier.
        img (str/ndarray): The image filename or loaded image.

    Returns:
        result (dict): The classification results that contains
            `class_name`, `pred_label` and `pred_score`.
    """
    cfg = model.cfg
    device = next(model.parameters()).device  # model device
    # build the data pipeline
    if isinstance(img, str):
        if cfg.data.test.pipeline[0]['type'] != 'LoadImageFromFile':
            cfg.data.test.pipeline.insert(0, dict(type='LoadImageFromFile'))
        data = dict(img_info=dict(filename=img), img_prefix=None)
    else:
        if cfg.data.test.pipeline[0]['type'] == 'LoadImageFromFile':
            cfg.data.test.pipeline.pop(0)
        data = dict(img=img)
    test_pipeline = Compose(cfg.data.test.pipeline)
    data = test_pipeline(data)
    data = collate([data], samples_per_gpu=1)
    if next(model.parameters()).is_cuda:
        # scatter to specified GPU
        data = scatter(data, [device])[0]

    # forward the model
    with torch.no_grad():
        scores = model(return_loss=False, **data)
        pred_score = np.max(scores, axis=1)[0]
        pred_label = np.argmax(scores, axis=1)[0]
        result = {'pred_label': pred_label, 'pred_score': float(pred_score)}
    result['pred_class'] = model.CLASSES[result['pred_label']]
    return result


def show_result_pyplot(model,
                       img,
                       result,
                       fig_size=(15, 10),
                       title='result',
                       wait_time=0):
    """Visualize the classification results on the image.

    Args:
        model (nn.Module): The loaded classifier.
        img (str or np.ndarray): Image filename or loaded image.
        result (list): The classification result.
        fig_size (tuple): Figure size of the pyplot figure.
            Defaults to (15, 10).
        title (str): Title of the pyplot figure.
            Defaults to 'result'.
        wait_time (int): How many seconds to display the image.
            Defaults to 0.
    """
    if hasattr(model, 'module'):
        model = model.module
    model.show_result(
        img,
        result,
        show=True,
        fig_size=fig_size,
        win_name=title,
        wait_time=wait_time)
