import os

import numpy as np

from .base_dataset import BaseDataset
from .builder import DATASETS


@DATASETS.register_module()
class ImageNet(BaseDataset):
    """`ImageNet <http://www.image-net.org>`_ Dataset.

    This implementation is modified from
    https://github.com/pytorch/vision/blob/master/torchvision/datasets/imagenet.py  # noqa: E501
    """

    IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif')

    def load_annotations(self):
        if self.ann_file is None:
            classes, class_to_idx = find_classes(self.data_prefix)
            samples = make_dataset(
                self.data_prefix, class_to_idx, extensions=self.IMG_EXTENSIONS)
            if len(samples) == 0:
                raise (RuntimeError('Found 0 files in subfolders of: '
                                    f'{self.data_prefix}. '
                                    'Supported extensions are: '
                                    f'{",".join(self.IMG_EXTENSIONS)}'))

            self.CLASSES = classes
            self.class_to_idx = class_to_idx
        elif isinstance(self.ann_file, str):
            with open(self.ann_file) as f:
                samples = [x.strip().split(' ') for x in f.readlines()]
        else:
            raise TypeError('ann_file must be a str or None')
        self.samples = samples

        data_infos = []
        for filename, gt_label in self.samples:
            info = {'img_prefix': self.data_prefix}
            info['img_info'] = {'filename': filename}
            info['gt_label'] = np.array(gt_label, dtype=np.int64)
            data_infos.append(info)
        return data_infos


def has_file_allowed_extension(filename, extensions):
    """Checks if a file is an allowed extension.

    Args:
        filename (string): path to a file

    Returns:
        bool: True if the filename ends with a known image extension
    """
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in extensions)


def find_classes(root):
    """Find classes by folders under a root.

    Args:
        root (string): root directory of folders

    Returns:
        classes (list): a list of class names
        class_to_idx (dict): the map from class name to class idx
    """
    classes = [
        d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))
    ]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx


def make_dataset(root, class_to_idx, extensions):
    """Make dataset by walking all images under a root.

    Args:
        root (string): root directory of folders
        class_to_idx (dict): the map from class name to class idx
        extensions (tuple): allowed extensions

    Returns:
        images (list): a list of tuple where each element is (image, label)
    """
    images = []
    root = os.path.expanduser(root)
    for class_name in sorted(os.listdir(root)):
        _dir = os.path.join(root, class_name)
        if not os.path.isdir(_dir):
            continue

        for _, _, fns in sorted(os.walk(_dir)):
            for fn in sorted(fns):
                if has_file_allowed_extension(fn, extensions):
                    path = os.path.join(class_name, fn)
                    item = (path, class_to_idx[class_name])
                    images.append(item)

    return images
