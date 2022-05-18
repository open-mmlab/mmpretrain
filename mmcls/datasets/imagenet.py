# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional, Union

from mmcls.registry import DATASETS
from mmcls.utils import get_root_logger
from .categories import IMAGENET_CATEGORIES
from .custom import CustomDataset


@DATASETS.register_module()
class ImageNet(CustomDataset):
    """`ImageNet <http://www.image-net.org>`_ Dataset.

    The dataset supports two kinds of annotation format. More details can be
    found in :class:`CustomDataset`.

    Args:
        ann_file (str, optional): Annotation file path. Defaults to None.
        metainfo (dict, optional): Meta information for dataset, such as class
            information. Defaults to None.
        data_root (str, optional): The root directory for ``data_prefix`` and
            ``ann_file``. Defaults to None.
        data_prefix (str | dict, optional): Prefix for training data. Defaults
            to None.
        **kwargs: Other keyword arguments in :class:`CustomDataset` and
            :class:`BaseDataset`.
    """  # noqa: E501

    IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif')
    METAINFO = {'CLASSES': IMAGENET_CATEGORIES}

    def __init__(self,
                 ann_file: Optional[str] = None,
                 metainfo: Optional[dict] = None,
                 data_root: Optional[str] = None,
                 data_prefix: Union[str, dict, None] = None,
                 **kwargs):
        kwargs = {'extensions': self.IMG_EXTENSIONS, **kwargs}
        super().__init__(
            ann_file=ann_file,
            metainfo=metainfo,
            data_root=data_root,
            data_prefix=data_prefix,
            **kwargs)


@DATASETS.register_module()
class ImageNet21k(CustomDataset):
    """ImageNet21k Dataset.

    Since the dataset ImageNet21k is extremely big, cantains 21k+ classes
    and 1.4B files. We won't provide the default categories list. Please
    specify it from the ``classes`` argument.

    Args:
        ann_file (str, optional): Annotation file path. Defaults to None.
        metainfo (dict, optional): Meta information for dataset, such as class
            information. Defaults to None.
        data_root (str, optional): The root directory for ``data_prefix`` and
            ``ann_file``. Defaults to None.
        data_prefix (str | dict, optional): Prefix for training data. Defaults
            to None.
        multi_label (bool): Not implement by now. Use multi label or not.
            Defaults to False.
        **kwargs: Other keyword arguments in :class:`CustomDataset` and
            :class:`BaseDataset`.
    """

    IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif')

    def __init__(self,
                 ann_file: Optional[str] = None,
                 metainfo: Optional[dict] = None,
                 data_root: Optional[str] = None,
                 data_prefix: Union[str, dict, None] = None,
                 multi_label: bool = False,
                 **kwargs):
        if multi_label:
            raise NotImplementedError(
                'The `multi_label` option is not supported by now.')
        self.multi_label = multi_label

        logger = get_root_logger()

        if ann_file is None:
            logger.warning(
                'The ImageNet21k dataset is large, and scanning directory may '
                'consume long time. Considering to specify the `ann_file` to '
                'accelerate the initialization.')

        kwargs = {'extensions': self.IMG_EXTENSIONS, **kwargs}
        super().__init__(
            ann_file=ann_file,
            metainfo=metainfo,
            data_root=data_root,
            data_prefix=data_prefix,
            **kwargs)

        if self.CLASSES is None:
            logger.warning(
                'The CLASSES is not stored in the `ImageNet21k` class. '
                'Considering to specify the `classes` argument if you need '
                'do inference on the ImageNet-21k dataset')
