# Copyright (c) OpenMMLab. All rights reserved.
from .inference import inference_model, init_model
from .list_registry import list_registry

__all__ = ['init_model', 'inference_model', 'list_registry']
