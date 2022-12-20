# Copyright (c) OpenMMLab. All rights reserved.
from typing import TYPE_CHECKING, Union

import numpy as np
import torch

if TYPE_CHECKING:
    from mmengine.model import BaseModel


def inference_model(model: 'BaseModel', img: Union[str, np.ndarray]):
    """Inference image(s) with the classifier.

    Args:
        model (BaseClassifier): The loaded classifier.
        img (str/ndarray): The image filename or loaded image.

    Returns:
        result (dict): The classification results that contains
            `class_name`, `pred_label` and `pred_score`.
    """
    from mmengine.dataset import Compose, default_collate
    from mmengine.registry import DefaultScope

    import mmcls.datasets  # noqa: F401

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
    with DefaultScope.overwrite_default_scope('mmcls'):
        test_pipeline = Compose(test_pipeline_cfg)
    data = test_pipeline(data)
    data = default_collate([data])

    # forward the model
    with torch.no_grad():
        prediction = model.val_step(data)[0].pred_label
        pred_scores = prediction.score.tolist()
        pred_score = torch.max(prediction.score).item()
        pred_label = torch.argmax(prediction.score).item()
        result = {
            'pred_label': pred_label,
            'pred_score': float(pred_score),
            'pred_scores': pred_scores
        }
    if hasattr(model, 'CLASSES'):
        result['pred_class'] = model.CLASSES[result['pred_label']]
    return result
