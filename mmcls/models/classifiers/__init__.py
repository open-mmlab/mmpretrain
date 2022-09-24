# Copyright (c) OpenMMLab. All rights reserved.
from .base import BaseClassifier, BaseRetriever
from .image import ImageClassifier
from .image2image import ImageToImageRetriever

__all__ = [
    'BaseClassifier', 'ImageClassifier', 'BaseRetriever',
    'ImageToImageRetriever'
]
