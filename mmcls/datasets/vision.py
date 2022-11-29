# Copyright (c) OpenMMLab. All rights reserved.
import io

import torch
from mmcv.transforms import BaseTransform
from mmcv.transforms import LoadImageFromFile as BaseLoadImageFromFile
from PIL import Image
from torchvision.transforms import autoaugment, transforms
from torchvision.transforms.functional import InterpolationMode

from mmcls.datasets import ImageNet
from mmcls.registry import DATASETS
from mmcls.structures import ClsDataSample


class PackVisionInputs(BaseTransform):
    """Pack the inputs data for the classification."""

    def __init__(self,
                 meta_keys=('sample_idx', 'img_path', 'ori_shape', 'img_shape',
                            'scale_factor', 'flip', 'flip_direction')):
        self.meta_keys = meta_keys

    def transform(self, results: dict) -> dict:
        """Method to pack the input data."""
        packed_results = dict()
        packed_results['inputs'] = results['img']

        data_sample = ClsDataSample()
        data_sample.set_gt_label(results['gt_label'])

        data_sample.set_metainfo(dict())
        packed_results['data_samples'] = data_sample

        return packed_results

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f'(meta_keys={self.meta_keys})'
        return repr_str


@DATASETS.register_module()
class VisionImageNet(ImageNet):
    """用以替换 `torchvision.datasets`  的 `ImageNet` 和 `ImageFolder`.
    Note:
        在具体使用时, 需要传入 transform 参数
    Example:
        >>> from dataset import MMClsImageNet
        >>> # 读本地的文件夹形式,
        >>> train = MMClsImageNet("./data/imagenet/train")
        >>> len(train)
        1281167
        >>> train[12131]
        (<PIL.Image.Image image mode=RGB size=500x333 at 0x7FDDAC778550>, 9)
        >>>
        >>> # 读 ceph 上的标注格式 (s 集群上默认格式)
        >>> val = MMClsImageNet(
        >>>    "./data/imagenet/val",
        >>>    ann_file="./data/imagenet/meta/val.txt",
        >>>    local=False,
        >>>    cluster_name ='openmmlab')
        >>> len(val)
        50000
        >>> val[1000]
        (<PIL.Image.Image image mode=RGB size=500x317 at 0x7FDDAC778430>, 188)
    """

    def __init__(self,
                 data_prefix,
                 ann_file='',
                 transform=None,
                 local=True,
                 cluster_name='openmmlab',
                 **kwargs):
        super().__init__(ann_file=ann_file, data_prefix=data_prefix, **kwargs)
        if 'train' in data_prefix:
            self.transform = create_transforms('train')
        elif 'val' in data_prefix or 'test' in data_prefix:
            self.transform = create_transforms('test')
        else:
            raise ValueError(f"{data_prefix} should contain 'train|val|test'")
        self.pack = PackVisionInputs()
        if local:
            file_client_args: dict = dict(backend='disk')
            self.load = PILLoadImageFromFile(file_client_args=file_client_args)
        else:
            file_client_args = dict(
                backend='petrel',
                path_mapping=dict({
                    './data/imagenet/':
                    f'{cluster_name}:s3://openmmlab/datasets/'
                    'classification/imagenet/',
                    'data/imagenet/':
                    f'{cluster_name}:s3://openmmlab/datasets/'
                    'classification/imagenet/'
                }))
            self.load = PILLoadImageFromFile(file_client_args=file_client_args)

    def __getitem__(self, idx: int) -> dict:
        result = super().__getitem__(idx)
        result = self.load(result)
        img, label = result['img'], result['gt_label']
        img = img.convert('RGB')
        try:
            if self.transform is not None:
                img = self.transform(img)
        except RuntimeError as e:
            print(result)
            raise e
        result = self.pack(dict(img=img, gt_label=int(label)))
        return result


class PILLoadImageFromFile(BaseLoadImageFromFile):

    def transform(self, results: dict):
        filename = results['img_path']
        try:
            img_bytes = self.file_client.get(filename)
            buff = io.BytesIO(img_bytes)
            results['img'] = Image.open(buff)

        except Exception as e:
            if self.ignore_empty:
                return None
            else:
                raise e

        results['img_shape'] = results['img'].size
        results['ori_shape'] = results['img'].size
        return results


def create_transforms(phase='train'):
    if phase == 'train':
        trans = ClassificationPresetTrain(crop_size=224, )
    else:
        trans = ClassificationPresetEval(
            crop_size=224,
            resize_size=256,
            interpolation=InterpolationMode.BILINEAR)
    return trans


class ClassificationPresetTrain:

    def __init__(
        self,
        *,
        crop_size,
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
        interpolation=InterpolationMode.BILINEAR,
        hflip_prob=0.5,
        auto_augment_policy=None,
        random_erase_prob=0.0,
    ):
        print(f'crop_size {crop_size}')
        print(f'mean {mean}')
        print(f'std {std}')
        print(f'interpolation {interpolation}')
        print(f'hflip_prob {hflip_prob}')
        print(f'auto_augment_policy {auto_augment_policy}')
        print(f'random_erase_prob {random_erase_prob}')
        trans = [
            transforms.RandomResizedCrop(
                crop_size, interpolation=interpolation)
        ]
        if hflip_prob > 0:
            trans.append(transforms.RandomHorizontalFlip(hflip_prob))
        if auto_augment_policy is not None:
            if auto_augment_policy == 'ra':
                trans.append(
                    autoaugment.RandAugment(interpolation=interpolation))
            elif auto_augment_policy == 'ta_wide':
                trans.append(
                    autoaugment.TrivialAugmentWide(
                        interpolation=interpolation))
            elif auto_augment_policy == 'augmix':
                trans.append(autoaugment.AugMix(interpolation=interpolation))
            else:
                aa_policy = autoaugment.AutoAugmentPolicy(auto_augment_policy)
                trans.append(
                    autoaugment.AutoAugment(
                        policy=aa_policy, interpolation=interpolation))
        trans.extend([
            transforms.PILToTensor(),
            transforms.ConvertImageDtype(torch.float),
            transforms.Normalize(mean=mean, std=std),
        ])
        if random_erase_prob > 0:
            trans.append(transforms.RandomErasing(p=random_erase_prob))

        self.transforms = transforms.Compose(trans)
        print(self.transforms)

    def __call__(self, img):
        return self.transforms(img)


'''
trans = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])])
'''


class ClassificationPresetEval:

    def __init__(
            self,
            *,
            crop_size,
            resize_size=256,
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
            interpolation=InterpolationMode.BILINEAR,
    ):
        print(f'crop_size {crop_size}')
        print(f'mean {mean}')
        print(f'std {std}')
        print(f'interpolation {interpolation}')

        self.transforms = transforms.Compose([
            transforms.Resize(resize_size, interpolation=interpolation),
            transforms.CenterCrop(crop_size),
            transforms.PILToTensor(),
            transforms.ConvertImageDtype(torch.float),
            transforms.Normalize(mean=mean, std=std),
        ])
        print(transforms)

    def __call__(self, img):
        return self.transforms(img)
