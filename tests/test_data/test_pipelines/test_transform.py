# Copyright (c) OpenMMLab. All rights reserved.
import copy
import os.path as osp
import random

import mmcv
import numpy as np
import pytest
import torch
import torchvision
from mmcv.utils import build_from_cfg
from numpy.testing import assert_array_almost_equal, assert_array_equal
from PIL import Image
from torchvision import transforms

import mmcls.datasets.pipelines.transforms as mmcls_transforms
from mmcls.datasets.builder import PIPELINES
from mmcls.datasets.pipelines import Compose


def construct_toy_data():
    img = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]],
                   dtype=np.uint8)
    img = np.stack([img, img, img], axis=-1)
    results = dict()
    # image
    results['ori_img'] = img
    results['img'] = copy.deepcopy(img)
    results['ori_shape'] = img.shape
    results['img_shape'] = img.shape
    return results


def test_resize():
    # test assertion if size is smaller than 0
    with pytest.raises(AssertionError):
        transform = dict(type='Resize', size=-1)
        build_from_cfg(transform, PIPELINES)

    # test assertion if size is tuple but the second value is smaller than 0
    # and the second value is not equal to -1
    with pytest.raises(AssertionError):
        transform = dict(type='Resize', size=(224, -2))
        build_from_cfg(transform, PIPELINES)

    # test assertion if size is tuple but the first value is smaller than 0
    with pytest.raises(AssertionError):
        transform = dict(type='Resize', size=(-1, 224))
        build_from_cfg(transform, PIPELINES)

    # test assertion if size is tuple and len(size) < 2
    with pytest.raises(AssertionError):
        transform = dict(type='Resize', size=(224, ))
        build_from_cfg(transform, PIPELINES)

    # test assertion if size is tuple len(size) > 2
    with pytest.raises(AssertionError):
        transform = dict(type='Resize', size=(224, 224, 3))
        build_from_cfg(transform, PIPELINES)

    # test assertion when interpolation is invalid
    with pytest.raises(AssertionError):
        transform = dict(type='Resize', size=224, interpolation='2333')
        build_from_cfg(transform, PIPELINES)

    # test assertion when resize_short is invalid
    with pytest.raises(AssertionError):
        transform = dict(type='Resize', size=224, adaptive_side='False')
        build_from_cfg(transform, PIPELINES)

    # test repr
    transform = dict(type='Resize', size=224)
    resize_module = build_from_cfg(transform, PIPELINES)
    assert isinstance(repr(resize_module), str)

    # read test image
    results = dict()
    img = mmcv.imread(
        osp.join(osp.dirname(__file__), '../../data/color.jpg'), 'color')
    original_img = copy.deepcopy(img)
    results['img'] = img
    results['img2'] = copy.deepcopy(img)
    results['img_shape'] = img.shape
    results['ori_shape'] = img.shape
    results['img_fields'] = ['img', 'img2']

    def reset_results(results, original_img):
        results['img'] = copy.deepcopy(original_img)
        results['img2'] = copy.deepcopy(original_img)
        results['img_shape'] = original_img.shape
        results['ori_shape'] = original_img.shape
        results['img_fields'] = ['img', 'img2']
        return results

    # test resize when size is int
    transform = dict(type='Resize', size=224, interpolation='bilinear')
    resize_module = build_from_cfg(transform, PIPELINES)
    results = resize_module(results)
    assert np.equal(results['img'], results['img2']).all()
    assert results['img_shape'] == (224, 224, 3)

    # test resize when size is tuple and the second value is -1
    transform = dict(type='Resize', size=(224, -1), interpolation='bilinear')
    resize_module = build_from_cfg(transform, PIPELINES)
    results = reset_results(results, original_img)
    results = resize_module(results)
    assert np.equal(results['img'], results['img2']).all()
    assert results['img_shape'] == (224, 298, 3)

    # test resize when size is tuple
    transform = dict(type='Resize', size=(224, 224), interpolation='bilinear')
    resize_module = build_from_cfg(transform, PIPELINES)
    results = reset_results(results, original_img)
    results = resize_module(results)
    assert np.equal(results['img'], results['img2']).all()
    assert results['img_shape'] == (224, 224, 3)

    # test resize when resize_height != resize_width
    transform = dict(type='Resize', size=(224, 256), interpolation='bilinear')
    resize_module = build_from_cfg(transform, PIPELINES)
    results = reset_results(results, original_img)
    results = resize_module(results)
    assert np.equal(results['img'], results['img2']).all()
    assert results['img_shape'] == (224, 256, 3)

    # test resize when size is larger than img.shape
    img_height, img_width, _ = original_img.shape
    transform = dict(
        type='Resize',
        size=(img_height * 2, img_width * 2),
        interpolation='bilinear')
    resize_module = build_from_cfg(transform, PIPELINES)
    results = reset_results(results, original_img)
    results = resize_module(results)
    assert np.equal(results['img'], results['img2']).all()
    assert results['img_shape'] == (img_height * 2, img_width * 2, 3)

    # test resize with different backends
    transform_cv2 = dict(
        type='Resize',
        size=(224, 256),
        interpolation='bilinear',
        backend='cv2')
    transform_pil = dict(
        type='Resize',
        size=(224, 256),
        interpolation='bilinear',
        backend='pillow')
    resize_module_cv2 = build_from_cfg(transform_cv2, PIPELINES)
    resize_module_pil = build_from_cfg(transform_pil, PIPELINES)
    results = reset_results(results, original_img)
    results['img_fields'] = ['img']
    results_cv2 = resize_module_cv2(results)
    results['img_fields'] = ['img2']
    results_pil = resize_module_pil(results)
    assert np.allclose(results_cv2['img'], results_pil['img2'], atol=45)

    # compare results with torchvision
    transform = dict(type='Resize', size=(224, 224), interpolation='area')
    resize_module = build_from_cfg(transform, PIPELINES)
    results = reset_results(results, original_img)
    results = resize_module(results)
    resize_module = transforms.Resize(
        size=(224, 224), interpolation=Image.BILINEAR)
    pil_img = Image.fromarray(original_img)
    resized_img = resize_module(pil_img)
    resized_img = np.array(resized_img)
    assert np.equal(results['img'], results['img2']).all()
    assert results['img_shape'] == (224, 224, 3)
    assert np.allclose(results['img'], resized_img, atol=30)

    # test resize when size is tuple, the second value is -1
    # and adaptive_side='long'
    transform = dict(
        type='Resize',
        size=(224, -1),
        adaptive_side='long',
        interpolation='bilinear')
    resize_module = build_from_cfg(transform, PIPELINES)
    results = reset_results(results, original_img)
    results = resize_module(results)
    assert np.equal(results['img'], results['img2']).all()
    assert results['img_shape'] == (168, 224, 3)

    # test resize when size is tuple, the second value is -1
    # and adaptive_side='long', h > w
    transform1 = dict(type='Resize', size=(300, 200), interpolation='bilinear')
    resize_module1 = build_from_cfg(transform1, PIPELINES)
    transform2 = dict(
        type='Resize',
        size=(224, -1),
        adaptive_side='long',
        interpolation='bilinear')
    resize_module2 = build_from_cfg(transform2, PIPELINES)
    results = reset_results(results, original_img)
    results = resize_module1(results)
    results = resize_module2(results)
    assert np.equal(results['img'], results['img2']).all()
    assert results['img_shape'] == (224, 149, 3)

    # test resize when size is tuple, the second value is -1
    # and adaptive_side='short', h > w
    transform1 = dict(type='Resize', size=(300, 200), interpolation='bilinear')
    resize_module1 = build_from_cfg(transform1, PIPELINES)
    transform2 = dict(
        type='Resize',
        size=(224, -1),
        adaptive_side='short',
        interpolation='bilinear')
    resize_module2 = build_from_cfg(transform2, PIPELINES)
    results = reset_results(results, original_img)
    results = resize_module1(results)
    results = resize_module2(results)
    assert np.equal(results['img'], results['img2']).all()
    assert results['img_shape'] == (336, 224, 3)

    # test interpolation method checking
    with pytest.raises(AssertionError):
        transform = dict(
            type='Resize', size=(300, 200), backend='cv2', interpolation='box')
        resize_module = build_from_cfg(transform, PIPELINES)

    with pytest.raises(AssertionError):
        transform = dict(
            type='Resize',
            size=(300, 200),
            backend='pillow',
            interpolation='area')
        resize_module = build_from_cfg(transform, PIPELINES)


def test_pad():
    results = dict()
    img = mmcv.imread(
        osp.join(osp.dirname(__file__), '../../data/color.jpg'), 'color')
    results['img'] = img
    results['img2'] = copy.deepcopy(img)
    results['img_shape'] = img.shape
    results['ori_shape'] = img.shape
    results['img_fields'] = ['img', 'img2']

    # test assertion if shape is None
    with pytest.raises(AssertionError):
        transform = dict(type='Pad', size=None)
        pad_module = build_from_cfg(transform, PIPELINES)
        pad_result = pad_module(copy.deepcopy(results))
        assert np.equal(pad_result['img'], pad_result['img2']).all()
        assert pad_result['img_shape'] == (400, 400, 3)

    # test if pad is valid
    transform = dict(type='Pad', size=(400, 400))
    pad_module = build_from_cfg(transform, PIPELINES)
    pad_result = pad_module(copy.deepcopy(results))
    assert isinstance(repr(pad_module), str)
    assert np.equal(pad_result['img'], pad_result['img2']).all()
    assert pad_result['img_shape'] == (400, 400, 3)
    assert np.allclose(pad_result['img'][-100:, :, :], 0)

    # test if pad_to_square is valid
    transform = dict(type='Pad', pad_to_square=True)
    pad_module = build_from_cfg(transform, PIPELINES)
    pad_result = pad_module(copy.deepcopy(results))
    assert isinstance(repr(pad_module), str)
    assert np.equal(pad_result['img'], pad_result['img2']).all()
    assert pad_result['img_shape'] == (400, 400, 3)
    assert np.allclose(pad_result['img'][-100:, :, :], 0)


def test_center_crop():
    # test assertion if size is smaller than 0
    with pytest.raises(AssertionError):
        transform = dict(type='CenterCrop', crop_size=-1)
        build_from_cfg(transform, PIPELINES)

    # test assertion if size is tuple but one value is smaller than 0
    with pytest.raises(AssertionError):
        transform = dict(type='CenterCrop', crop_size=(224, -1))
        build_from_cfg(transform, PIPELINES)

    # test assertion if size is tuple and len(size) < 2
    with pytest.raises(AssertionError):
        transform = dict(type='CenterCrop', crop_size=(224, ))
        build_from_cfg(transform, PIPELINES)

    # test assertion if size is tuple len(size) > 2
    with pytest.raises(AssertionError):
        transform = dict(type='CenterCrop', crop_size=(224, 224, 3))
        build_from_cfg(transform, PIPELINES)

    # test assertion if efficientnet is True and crop_size is tuple
    with pytest.raises(AssertionError):
        transform = dict(
            type='CenterCrop',
            crop_size=(224, 224),
            efficientnet_style=True,
        )
        build_from_cfg(transform, PIPELINES)

    # test assertion if efficientnet is True and interpolation is invalid
    with pytest.raises(AssertionError):
        transform = dict(
            type='CenterCrop',
            crop_size=224,
            efficientnet_style=True,
            interpolation='2333')
        build_from_cfg(transform, PIPELINES)

    # test assertion if efficientnet is True and crop_padding is negative
    with pytest.raises(AssertionError):
        transform = dict(
            type='CenterCrop',
            crop_size=224,
            efficientnet_style=True,
            crop_padding=-1)
        build_from_cfg(transform, PIPELINES)

    # test repr
    transform = dict(type='CenterCrop', crop_size=224)
    center_crop_module = build_from_cfg(transform, PIPELINES)
    assert isinstance(repr(center_crop_module), str)

    # read test image
    results = dict()
    img = mmcv.imread(
        osp.join(osp.dirname(__file__), '../../data/color.jpg'), 'color')
    original_img = copy.deepcopy(img)
    results['img'] = img
    results['img2'] = copy.deepcopy(img)
    results['img_shape'] = img.shape
    results['ori_shape'] = img.shape
    results['img_fields'] = ['img', 'img2']

    def reset_results(results, original_img):
        results['img'] = copy.deepcopy(original_img)
        results['img2'] = copy.deepcopy(original_img)
        results['img_shape'] = original_img.shape
        results['ori_shape'] = original_img.shape
        return results

    # test CenterCrop when size is int
    transform = dict(type='CenterCrop', crop_size=224)
    center_crop_module = build_from_cfg(transform, PIPELINES)
    results = center_crop_module(results)
    assert np.equal(results['img'], results['img2']).all()
    assert results['img_shape'] == (224, 224, 3)

    # test CenterCrop when size is int and efficientnet_style is True
    # and crop_padding=0
    transform = dict(
        type='CenterCrop',
        crop_size=224,
        efficientnet_style=True,
        crop_padding=0)
    center_crop_module = build_from_cfg(transform, PIPELINES)
    results = reset_results(results, original_img)
    results = center_crop_module(results)
    assert np.equal(results['img'], results['img2']).all()
    assert results['img_shape'] == (224, 224, 3)
    results_img = copy.deepcopy(results['img'])

    short_edge = min(*results['ori_shape'][:2])
    transform = dict(type='CenterCrop', crop_size=short_edge)
    baseline_center_crop_module = build_from_cfg(transform, PIPELINES)
    transform = dict(type='Resize', size=224)
    baseline_resize_module = build_from_cfg(transform, PIPELINES)
    results = reset_results(results, original_img)
    results = baseline_center_crop_module(results)
    results = baseline_resize_module(results)
    assert np.equal(results['img'], results_img).all()

    # test CenterCrop when size is tuple
    transform = dict(type='CenterCrop', crop_size=(224, 224))
    center_crop_module = build_from_cfg(transform, PIPELINES)
    results = reset_results(results, original_img)
    results = center_crop_module(results)
    assert np.equal(results['img'], results['img2']).all()
    assert results['img_shape'] == (224, 224, 3)

    # test CenterCrop when crop_height != crop_width
    transform = dict(type='CenterCrop', crop_size=(256, 224))
    center_crop_module = build_from_cfg(transform, PIPELINES)
    results = reset_results(results, original_img)
    results = center_crop_module(results)
    assert np.equal(results['img'], results['img2']).all()
    assert results['img_shape'] == (256, 224, 3)

    # test CenterCrop when crop_size is equal to img.shape
    img_height, img_width, _ = original_img.shape
    transform = dict(type='CenterCrop', crop_size=(img_height, img_width))
    center_crop_module = build_from_cfg(transform, PIPELINES)
    results = reset_results(results, original_img)
    results = center_crop_module(results)
    assert np.equal(results['img'], results['img2']).all()
    assert results['img_shape'] == (img_height, img_width, 3)

    # test CenterCrop when crop_size is larger than img.shape
    transform = dict(
        type='CenterCrop', crop_size=(img_height * 2, img_width * 2))
    center_crop_module = build_from_cfg(transform, PIPELINES)
    results = reset_results(results, original_img)
    results = center_crop_module(results)
    assert np.equal(results['img'], results['img2']).all()
    assert results['img_shape'] == (img_height, img_width, 3)

    # test CenterCrop when crop_width is smaller than img_width
    transform = dict(type='CenterCrop', crop_size=(img_height, img_width / 2))
    center_crop_module = build_from_cfg(transform, PIPELINES)
    results = reset_results(results, original_img)
    results = center_crop_module(results)
    assert np.equal(results['img'], results['img2']).all()
    assert results['img_shape'] == (img_height, img_width / 2, 3)

    # test CenterCrop when crop_height is smaller than img_height
    transform = dict(type='CenterCrop', crop_size=(img_height / 2, img_width))
    center_crop_module = build_from_cfg(transform, PIPELINES)
    results = reset_results(results, original_img)
    results = center_crop_module(results)
    assert np.equal(results['img'], results['img2']).all()
    assert results['img_shape'] == (img_height / 2, img_width, 3)

    # compare results with torchvision
    transform = dict(type='CenterCrop', crop_size=224)
    center_crop_module = build_from_cfg(transform, PIPELINES)
    results = reset_results(results, original_img)
    results = center_crop_module(results)
    center_crop_module = transforms.CenterCrop(size=224)
    pil_img = Image.fromarray(original_img)
    cropped_img = center_crop_module(pil_img)
    cropped_img = np.array(cropped_img)
    assert np.equal(results['img'], results['img2']).all()
    assert np.equal(results['img'], cropped_img).all()


def test_normalize():
    img_norm_cfg = dict(
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True)

    # test repr
    transform = dict(type='Normalize', **img_norm_cfg)
    normalize_module = build_from_cfg(transform, PIPELINES)
    assert isinstance(repr(normalize_module), str)

    # read data
    results = dict()
    img = mmcv.imread(
        osp.join(osp.dirname(__file__), '../../data/color.jpg'), 'color')
    original_img = copy.deepcopy(img)
    results['img'] = img
    results['img2'] = copy.deepcopy(img)
    results['img_shape'] = img.shape
    results['ori_shape'] = img.shape
    results['img_fields'] = ['img', 'img2']

    norm_results = normalize_module(results)
    assert np.equal(norm_results['img'], norm_results['img2']).all()

    # compare results with manual computation
    mean = np.array(img_norm_cfg['mean'])
    std = np.array(img_norm_cfg['std'])
    normalized_img = (original_img[..., ::-1] - mean) / std
    assert np.allclose(norm_results['img'], normalized_img)

    # compare results with torchvision
    normalize_module = transforms.Normalize(mean=mean, std=std)
    tensor_img = original_img[..., ::-1].copy()
    tensor_img = torch.Tensor(tensor_img.transpose(2, 0, 1))
    normalized_img = normalize_module(tensor_img)
    normalized_img = np.array(normalized_img).transpose(1, 2, 0)
    assert np.equal(norm_results['img'], normalized_img).all()


def test_randomcrop():
    ori_img = mmcv.imread(
        osp.join(osp.dirname(__file__), '../../data/color.jpg'), 'color')
    ori_img_pil = Image.open(
        osp.join(osp.dirname(__file__), '../../data/color.jpg'))
    seed = random.randint(0, 100)

    # test crop size is int
    kwargs = dict(size=200, padding=0, pad_if_needed=True, fill=0)
    random.seed(seed)
    np.random.seed(seed)
    aug = []
    aug.extend([torchvision.transforms.RandomCrop(**kwargs)])
    composed_transform = Compose(aug)
    baseline = composed_transform(ori_img_pil)

    kwargs = dict(size=200, padding=0, pad_if_needed=True, pad_val=0)
    random.seed(seed)
    np.random.seed(seed)
    aug = []
    aug.extend([mmcls_transforms.RandomCrop(**kwargs)])
    composed_transform = Compose(aug)

    # test __repr__()
    print(composed_transform)
    results = dict()
    results['img'] = ori_img
    img = composed_transform(results)['img']
    assert np.array(img).shape == (200, 200, 3)
    assert np.array(baseline).shape == (200, 200, 3)
    nonzero = len((ori_img - np.array(ori_img_pil)[:, :, ::-1]).nonzero())
    nonzero_transform = len((img - np.array(baseline)[:, :, ::-1]).nonzero())
    assert nonzero == nonzero_transform

    # test crop size < image size
    kwargs = dict(size=(200, 300), padding=0, pad_if_needed=True, fill=0)
    random.seed(seed)
    np.random.seed(seed)
    aug = []
    aug.extend([torchvision.transforms.RandomCrop(**kwargs)])
    composed_transform = Compose(aug)
    baseline = composed_transform(ori_img_pil)

    kwargs = dict(size=(200, 300), padding=0, pad_if_needed=True, pad_val=0)
    random.seed(seed)
    np.random.seed(seed)
    aug = []
    aug.extend([mmcls_transforms.RandomCrop(**kwargs)])
    composed_transform = Compose(aug)
    results = dict()
    results['img'] = ori_img
    img = composed_transform(results)['img']
    assert np.array(img).shape == (200, 300, 3)
    assert np.array(baseline).shape == (200, 300, 3)
    nonzero = len((ori_img - np.array(ori_img_pil)[:, :, ::-1]).nonzero())
    nonzero_transform = len((img - np.array(baseline)[:, :, ::-1]).nonzero())
    assert nonzero == nonzero_transform

    # test crop size > image size
    kwargs = dict(size=(600, 700), padding=0, pad_if_needed=True, fill=0)
    random.seed(seed)
    np.random.seed(seed)
    aug = []
    aug.extend([torchvision.transforms.RandomCrop(**kwargs)])
    composed_transform = Compose(aug)
    baseline = composed_transform(ori_img_pil)

    kwargs = dict(size=(600, 700), padding=0, pad_if_needed=True, pad_val=0)
    random.seed(seed)
    np.random.seed(seed)
    aug = []
    aug.extend([mmcls_transforms.RandomCrop(**kwargs)])
    composed_transform = Compose(aug)
    results = dict()
    results['img'] = ori_img
    img = composed_transform(results)['img']
    assert np.array(img).shape == (600, 700, 3)
    assert np.array(baseline).shape == (600, 700, 3)
    nonzero = len((ori_img - np.array(ori_img_pil)[:, :, ::-1]).nonzero())
    nonzero_transform = len((img - np.array(baseline)[:, :, ::-1]).nonzero())
    assert nonzero == nonzero_transform

    # test crop size == image size
    kwargs = dict(
        size=(ori_img.shape[0], ori_img.shape[1]),
        padding=0,
        pad_if_needed=True,
        fill=0)
    random.seed(seed)
    np.random.seed(seed)
    aug = []
    aug.extend([torchvision.transforms.RandomCrop(**kwargs)])
    composed_transform = Compose(aug)
    baseline = composed_transform(ori_img_pil)

    kwargs = dict(
        size=(ori_img.shape[0], ori_img.shape[1]),
        padding=0,
        pad_if_needed=True,
        pad_val=0)
    random.seed(seed)
    np.random.seed(seed)
    aug = []
    aug.extend([mmcls_transforms.RandomCrop(**kwargs)])
    composed_transform = Compose(aug)
    results = dict()
    results['img'] = ori_img
    img = composed_transform(results)['img']

    assert np.array(img).shape == (img.shape[0], img.shape[1], 3)
    assert np.array(baseline).shape == (img.shape[0], img.shape[1], 3)
    nonzero = len((ori_img - np.array(ori_img_pil)[:, :, ::-1]).nonzero())
    nonzero_transform = len((img - np.array(baseline)[:, :, ::-1]).nonzero())
    assert nonzero == nonzero_transform
    assert_array_equal(ori_img, img)
    assert_array_equal(np.array(baseline), np.array(ori_img_pil))

    # test different padding mode
    for mode in ['constant', 'edge', 'reflect', 'symmetric']:
        kwargs = dict(size=(500, 600), padding=0, pad_if_needed=True, fill=0)
        kwargs['padding_mode'] = mode
        random.seed(seed)
        np.random.seed(seed)
        aug = []
        aug.extend([torchvision.transforms.RandomCrop(**kwargs)])
        composed_transform = Compose(aug)
        baseline = composed_transform(ori_img_pil)

        kwargs = dict(
            size=(500, 600), padding=0, pad_if_needed=True, pad_val=0)
        random.seed(seed)
        np.random.seed(seed)
        aug = []
        aug.extend([mmcls_transforms.RandomCrop(**kwargs)])
        composed_transform = Compose(aug)
        results = dict()
        results['img'] = ori_img
        img = composed_transform(results)['img']
        assert np.array(img).shape == (500, 600, 3)
        assert np.array(baseline).shape == (500, 600, 3)
        nonzero = len((ori_img - np.array(ori_img_pil)[:, :, ::-1]).nonzero())
        nonzero_transform = len(
            (img - np.array(baseline)[:, :, ::-1]).nonzero())
        assert nonzero == nonzero_transform


def test_randomresizedcrop():
    ori_img = mmcv.imread(
        osp.join(osp.dirname(__file__), '../../data/color.jpg'), 'color')
    ori_img_pil = Image.open(
        osp.join(osp.dirname(__file__), '../../data/color.jpg'))

    seed = random.randint(0, 100)

    # test when scale is not of kind (min, max)
    with pytest.raises(ValueError):
        kwargs = dict(
            size=(200, 300), scale=(1.0, 0.08), ratio=(3. / 4., 4. / 3.))
        aug = []
        aug.extend([mmcls_transforms.RandomResizedCrop(**kwargs)])
        composed_transform = Compose(aug)
        results = dict()
        results['img'] = ori_img
        composed_transform(results)['img']

    # test when ratio is not of kind (min, max)
    with pytest.raises(ValueError):
        kwargs = dict(
            size=(200, 300), scale=(0.08, 1.0), ratio=(4. / 3., 3. / 4.))
        aug = []
        aug.extend([mmcls_transforms.RandomResizedCrop(**kwargs)])
        composed_transform = Compose(aug)
        results = dict()
        results['img'] = ori_img
        composed_transform(results)['img']

    # test when efficientnet_style is True and crop_padding < 0
    with pytest.raises(AssertionError):
        kwargs = dict(size=200, efficientnet_style=True, crop_padding=-1)
        aug = []
        aug.extend([mmcls_transforms.RandomResizedCrop(**kwargs)])
        composed_transform = Compose(aug)
        results = dict()
        results['img'] = ori_img
        composed_transform(results)['img']

    # test crop size is int
    kwargs = dict(size=200, scale=(0.08, 1.0), ratio=(3. / 4., 4. / 3.))
    random.seed(seed)
    np.random.seed(seed)
    aug = []
    aug.extend([torchvision.transforms.RandomResizedCrop(**kwargs)])
    composed_transform = Compose(aug)
    baseline = composed_transform(ori_img_pil)

    random.seed(seed)
    np.random.seed(seed)
    aug = []
    aug.extend([mmcls_transforms.RandomResizedCrop(**kwargs)])
    composed_transform = Compose(aug)

    # test __repr__()
    print(composed_transform)
    results = dict()
    results['img'] = ori_img
    img = composed_transform(results)['img']
    assert np.array(img).shape == (200, 200, 3)
    assert np.array(baseline).shape == (200, 200, 3)
    nonzero = len((ori_img - np.array(ori_img_pil)[:, :, ::-1]).nonzero())
    nonzero_transform = len((img - np.array(baseline)[:, :, ::-1]).nonzero())
    assert nonzero == nonzero_transform

    # test crop size < image size
    kwargs = dict(size=(200, 300), scale=(0.08, 1.0), ratio=(3. / 4., 4. / 3.))
    random.seed(seed)
    np.random.seed(seed)
    aug = []
    aug.extend([torchvision.transforms.RandomResizedCrop(**kwargs)])
    composed_transform = Compose(aug)
    baseline = composed_transform(ori_img_pil)

    random.seed(seed)
    np.random.seed(seed)
    aug = []
    aug.extend([mmcls_transforms.RandomResizedCrop(**kwargs)])
    composed_transform = Compose(aug)
    results = dict()
    results['img'] = ori_img
    img = composed_transform(results)['img']
    assert np.array(img).shape == (200, 300, 3)
    assert np.array(baseline).shape == (200, 300, 3)
    nonzero = len((ori_img - np.array(ori_img_pil)[:, :, ::-1]).nonzero())
    nonzero_transform = len((img - np.array(baseline)[:, :, ::-1]).nonzero())
    assert nonzero == nonzero_transform

    # test crop size < image size when efficientnet_style = True
    kwargs = dict(
        size=200,
        scale=(0.08, 1.0),
        ratio=(3. / 4., 4. / 3.),
        efficientnet_style=True)
    random.seed(seed)
    np.random.seed(seed)
    aug = []
    aug.extend([mmcls_transforms.RandomResizedCrop(**kwargs)])
    composed_transform = Compose(aug)
    results = dict()
    results['img'] = ori_img
    img = composed_transform(results)['img']
    assert img.shape == (200, 200, 3)

    # test crop size > image size
    kwargs = dict(size=(600, 700), scale=(0.08, 1.0), ratio=(3. / 4., 4. / 3.))
    random.seed(seed)
    np.random.seed(seed)
    aug = []
    aug.extend([torchvision.transforms.RandomResizedCrop(**kwargs)])
    composed_transform = Compose(aug)
    baseline = composed_transform(ori_img_pil)

    random.seed(seed)
    np.random.seed(seed)
    aug = []
    aug.extend([mmcls_transforms.RandomResizedCrop(**kwargs)])
    composed_transform = Compose(aug)
    results = dict()
    results['img'] = ori_img
    img = composed_transform(results)['img']
    assert np.array(img).shape == (600, 700, 3)
    assert np.array(baseline).shape == (600, 700, 3)
    nonzero = len((ori_img - np.array(ori_img_pil)[:, :, ::-1]).nonzero())
    nonzero_transform = len((img - np.array(baseline)[:, :, ::-1]).nonzero())
    assert nonzero == nonzero_transform

    # test crop size < image size when efficientnet_style = True
    kwargs = dict(
        size=600,
        scale=(0.08, 1.0),
        ratio=(3. / 4., 4. / 3.),
        efficientnet_style=True)
    random.seed(seed)
    np.random.seed(seed)
    aug = []
    aug.extend([mmcls_transforms.RandomResizedCrop(**kwargs)])
    composed_transform = Compose(aug)
    results = dict()
    results['img'] = ori_img
    img = composed_transform(results)['img']
    assert img.shape == (600, 600, 3)

    # test cropping the whole image
    kwargs = dict(
        size=(ori_img.shape[0], ori_img.shape[1]),
        scale=(1.0, 2.0),
        ratio=(1.0, 2.0))
    random.seed(seed)
    np.random.seed(seed)
    aug = []
    aug.extend([torchvision.transforms.RandomResizedCrop(**kwargs)])
    composed_transform = Compose(aug)
    baseline = composed_transform(ori_img_pil)

    random.seed(seed)
    np.random.seed(seed)
    aug = []
    aug.extend([mmcls_transforms.RandomResizedCrop(**kwargs)])
    composed_transform = Compose(aug)
    results = dict()
    results['img'] = ori_img
    img = composed_transform(results)['img']
    assert np.array(img).shape == (ori_img.shape[0], ori_img.shape[1], 3)
    assert np.array(baseline).shape == (ori_img.shape[0], ori_img.shape[1], 3)
    nonzero = len((ori_img - np.array(ori_img_pil)[:, :, ::-1]).nonzero())
    nonzero_transform = len((img - np.array(baseline)[:, :, ::-1]).nonzero())
    assert nonzero == nonzero_transform
    # assert_array_equal(ori_img, img)
    # assert_array_equal(np.array(ori_img_pil), np.array(baseline))

    # test central crop when in_ratio < min(ratio)
    kwargs = dict(
        size=(ori_img.shape[0], ori_img.shape[1]),
        scale=(1.0, 2.0),
        ratio=(2., 3.))
    random.seed(seed)
    np.random.seed(seed)
    aug = []
    aug.extend([torchvision.transforms.RandomResizedCrop(**kwargs)])
    composed_transform = Compose(aug)
    baseline = composed_transform(ori_img_pil)

    random.seed(seed)
    np.random.seed(seed)
    aug = []
    aug.extend([mmcls_transforms.RandomResizedCrop(**kwargs)])
    composed_transform = Compose(aug)
    results = dict()
    results['img'] = ori_img
    img = composed_transform(results)['img']
    assert np.array(img).shape == (ori_img.shape[0], ori_img.shape[1], 3)
    assert np.array(baseline).shape == (ori_img.shape[0], ori_img.shape[1], 3)
    nonzero = len((ori_img - np.array(ori_img_pil)[:, :, ::-1]).nonzero())
    nonzero_transform = len((img - np.array(baseline)[:, :, ::-1]).nonzero())
    assert nonzero == nonzero_transform

    # test central crop when in_ratio > max(ratio)
    kwargs = dict(
        size=(ori_img.shape[0], ori_img.shape[1]),
        scale=(1.0, 2.0),
        ratio=(3. / 4., 1))
    random.seed(seed)
    np.random.seed(seed)
    aug = []
    aug.extend([torchvision.transforms.RandomResizedCrop(**kwargs)])
    composed_transform = Compose(aug)
    baseline = composed_transform(ori_img_pil)

    random.seed(seed)
    np.random.seed(seed)
    aug = []
    aug.extend([mmcls_transforms.RandomResizedCrop(**kwargs)])
    composed_transform = Compose(aug)
    results = dict()
    results['img'] = ori_img
    img = composed_transform(results)['img']
    assert np.array(img).shape == (ori_img.shape[0], ori_img.shape[1], 3)
    assert np.array(baseline).shape == (ori_img.shape[0], ori_img.shape[1], 3)
    nonzero = len((ori_img - np.array(ori_img_pil)[:, :, ::-1]).nonzero())
    nonzero_transform = len((img - np.array(baseline)[:, :, ::-1]).nonzero())
    assert nonzero == nonzero_transform

    # test central crop when max_attempts = 0 and efficientnet_style = True
    kwargs = dict(
        size=200,
        scale=(0.08, 1.0),
        ratio=(3. / 4., 4. / 3.),
        efficientnet_style=True,
        max_attempts=0,
        crop_padding=32)
    random.seed(seed)
    np.random.seed(seed)
    aug = []
    aug.extend([mmcls_transforms.RandomResizedCrop(**kwargs)])
    composed_transform = Compose(aug)
    results = dict()
    results['img'] = ori_img
    img = composed_transform(results)['img']

    kwargs = dict(crop_size=200, efficientnet_style=True, crop_padding=32)
    resize_kwargs = dict(size=200)
    random.seed(seed)
    np.random.seed(seed)
    aug = []
    aug.extend([mmcls_transforms.CenterCrop(**kwargs)])
    aug.extend([mmcls_transforms.Resize(**resize_kwargs)])
    composed_transform = Compose(aug)
    results = dict()
    results['img'] = ori_img
    baseline = composed_transform(results)['img']

    assert img.shape == baseline.shape
    assert np.equal(img, baseline).all()

    # test central crop when max_attempts = 0 and efficientnet_style = True
    kwargs = dict(
        size=200,
        scale=(0.08, 1.0),
        ratio=(3. / 4., 4. / 3.),
        efficientnet_style=True,
        max_attempts=100,
        min_covered=1)
    random.seed(seed)
    np.random.seed(seed)
    aug = []
    aug.extend([mmcls_transforms.RandomResizedCrop(**kwargs)])
    composed_transform = Compose(aug)
    results = dict()
    results['img'] = ori_img
    img = composed_transform(results)['img']

    kwargs = dict(crop_size=200, efficientnet_style=True, crop_padding=32)
    resize_kwargs = dict(size=200)
    random.seed(seed)
    np.random.seed(seed)
    aug = []
    aug.extend([mmcls_transforms.CenterCrop(**kwargs)])
    aug.extend([mmcls_transforms.Resize(**resize_kwargs)])
    composed_transform = Compose(aug)
    results = dict()
    results['img'] = ori_img
    baseline = composed_transform(results)['img']

    assert img.shape == baseline.shape
    assert np.equal(img, baseline).all()

    # test different interpolation types
    for mode in ['nearest', 'bilinear', 'bicubic', 'area', 'lanczos']:
        kwargs = dict(
            size=(600, 700),
            scale=(0.08, 1.0),
            ratio=(3. / 4., 4. / 3.),
            interpolation=mode)
        aug = []
        aug.extend([mmcls_transforms.RandomResizedCrop(**kwargs)])
        composed_transform = Compose(aug)
        results = dict()
        results['img'] = ori_img
        img = composed_transform(results)['img']
        assert img.shape == (600, 700, 3)


def test_randomgrayscale():

    # test rgb2gray, return the grayscale image with p>1
    in_img = np.random.rand(10, 10, 3).astype(np.float32)
    kwargs = dict(gray_prob=2)

    aug = []
    aug.extend([mmcls_transforms.RandomGrayscale(**kwargs)])
    composed_transform = Compose(aug)
    print(composed_transform)
    results = dict()
    results['img'] = in_img
    img = composed_transform(results)['img']
    computed_gray = (
        in_img[:, :, 0] * 0.299 + in_img[:, :, 1] * 0.587 +
        in_img[:, :, 2] * 0.114)
    for i in range(img.shape[2]):
        assert_array_almost_equal(img[:, :, i], computed_gray, decimal=4)
    assert img.shape == (10, 10, 3)

    # test rgb2gray, return the original image with p=-1
    in_img = np.random.rand(10, 10, 3).astype(np.float32)
    kwargs = dict(gray_prob=-1)

    aug = []
    aug.extend([mmcls_transforms.RandomGrayscale(**kwargs)])
    composed_transform = Compose(aug)
    results = dict()
    results['img'] = in_img
    img = composed_transform(results)['img']
    assert_array_equal(img, in_img)
    assert img.shape == (10, 10, 3)

    # test image with one channel with our method
    # and the function from torchvision
    in_img = np.random.rand(10, 10, 1).astype(np.float32)
    kwargs = dict(gray_prob=2)

    aug = []
    aug.extend([mmcls_transforms.RandomGrayscale(**kwargs)])
    composed_transform = Compose(aug)
    results = dict()
    results['img'] = in_img
    img = composed_transform(results)['img']
    assert_array_equal(img, in_img)
    assert img.shape == (10, 10, 1)

    in_img_pil = Image.fromarray(in_img[:, :, 0], mode='L')
    kwargs = dict(p=2)
    aug = []
    aug.extend([torchvision.transforms.RandomGrayscale(**kwargs)])
    composed_transform = Compose(aug)
    img_pil = composed_transform(in_img_pil)
    assert_array_equal(np.array(img_pil), np.array(in_img_pil))
    assert np.array(img_pil).shape == (10, 10)


def test_randomflip():
    # test assertion if flip probability is smaller than 0
    with pytest.raises(AssertionError):
        transform = dict(type='RandomFlip', flip_prob=-1)
        build_from_cfg(transform, PIPELINES)

    # test assertion if flip probability is larger than 1
    with pytest.raises(AssertionError):
        transform = dict(type='RandomFlip', flip_prob=2)
        build_from_cfg(transform, PIPELINES)

    # test assertion if direction is not horizontal and vertical
    with pytest.raises(AssertionError):
        transform = dict(type='RandomFlip', direction='random')
        build_from_cfg(transform, PIPELINES)

    # test assertion if direction is not lowercase
    with pytest.raises(AssertionError):
        transform = dict(type='RandomFlip', direction='Horizontal')
        build_from_cfg(transform, PIPELINES)

    # read test image
    results = dict()
    img = mmcv.imread(
        osp.join(osp.dirname(__file__), '../../data/color.jpg'), 'color')
    original_img = copy.deepcopy(img)
    results['img'] = img
    results['img2'] = copy.deepcopy(img)
    results['img_shape'] = img.shape
    results['ori_shape'] = img.shape
    results['img_fields'] = ['img', 'img2']

    def reset_results(results, original_img):
        results['img'] = copy.deepcopy(original_img)
        results['img2'] = copy.deepcopy(original_img)
        results['img_shape'] = original_img.shape
        results['ori_shape'] = original_img.shape
        return results

    # test RandomFlip when flip_prob is 0
    transform = dict(type='RandomFlip', flip_prob=0)
    flip_module = build_from_cfg(transform, PIPELINES)
    results = flip_module(results)
    assert np.equal(results['img'], original_img).all()
    assert np.equal(results['img'], results['img2']).all()

    # test RandomFlip when flip_prob is 1
    transform = dict(type='RandomFlip', flip_prob=1)
    flip_module = build_from_cfg(transform, PIPELINES)
    results = flip_module(results)
    assert np.equal(results['img'], results['img2']).all()

    # compare horizontal flip with torchvision
    transform = dict(type='RandomFlip', flip_prob=1, direction='horizontal')
    flip_module = build_from_cfg(transform, PIPELINES)
    results = reset_results(results, original_img)
    results = flip_module(results)
    flip_module = transforms.RandomHorizontalFlip(p=1)
    pil_img = Image.fromarray(original_img)
    flipped_img = flip_module(pil_img)
    flipped_img = np.array(flipped_img)
    assert np.equal(results['img'], results['img2']).all()
    assert np.equal(results['img'], flipped_img).all()

    # compare vertical flip with torchvision
    transform = dict(type='RandomFlip', flip_prob=1, direction='vertical')
    flip_module = build_from_cfg(transform, PIPELINES)
    results = reset_results(results, original_img)
    results = flip_module(results)
    flip_module = transforms.RandomVerticalFlip(p=1)
    pil_img = Image.fromarray(original_img)
    flipped_img = flip_module(pil_img)
    flipped_img = np.array(flipped_img)
    assert np.equal(results['img'], results['img2']).all()
    assert np.equal(results['img'], flipped_img).all()


def test_random_erasing():
    # test erase_prob assertion
    with pytest.raises(AssertionError):
        cfg = dict(type='RandomErasing', erase_prob=-1.)
        build_from_cfg(cfg, PIPELINES)
    with pytest.raises(AssertionError):
        cfg = dict(type='RandomErasing', erase_prob=1)
        build_from_cfg(cfg, PIPELINES)

    # test area_ratio assertion
    with pytest.raises(AssertionError):
        cfg = dict(type='RandomErasing', min_area_ratio=-1.)
        build_from_cfg(cfg, PIPELINES)
    with pytest.raises(AssertionError):
        cfg = dict(type='RandomErasing', max_area_ratio=1)
        build_from_cfg(cfg, PIPELINES)
    with pytest.raises(AssertionError):
        # min_area_ratio should be smaller than max_area_ratio
        cfg = dict(
            type='RandomErasing', min_area_ratio=0.6, max_area_ratio=0.4)
        build_from_cfg(cfg, PIPELINES)

    # test aspect_range assertion
    with pytest.raises(AssertionError):
        cfg = dict(type='RandomErasing', aspect_range='str')
        build_from_cfg(cfg, PIPELINES)
    with pytest.raises(AssertionError):
        cfg = dict(type='RandomErasing', aspect_range=-1)
        build_from_cfg(cfg, PIPELINES)
    with pytest.raises(AssertionError):
        # In aspect_range (min, max), min should be smaller than max.
        cfg = dict(type='RandomErasing', aspect_range=[1.6, 0.6])
        build_from_cfg(cfg, PIPELINES)

    # test mode assertion
    with pytest.raises(AssertionError):
        cfg = dict(type='RandomErasing', mode='unknown')
        build_from_cfg(cfg, PIPELINES)

    # test fill_std assertion
    with pytest.raises(AssertionError):
        cfg = dict(type='RandomErasing', fill_std='unknown')
        build_from_cfg(cfg, PIPELINES)

    # test implicit conversion of aspect_range
    cfg = dict(type='RandomErasing', aspect_range=0.5)
    random_erasing = build_from_cfg(cfg, PIPELINES)
    assert random_erasing.aspect_range == (0.5, 2.)

    cfg = dict(type='RandomErasing', aspect_range=2.)
    random_erasing = build_from_cfg(cfg, PIPELINES)
    assert random_erasing.aspect_range == (0.5, 2.)

    # test implicit conversion of fill_color
    cfg = dict(type='RandomErasing', fill_color=15)
    random_erasing = build_from_cfg(cfg, PIPELINES)
    assert random_erasing.fill_color == [15, 15, 15]

    # test implicit conversion of fill_std
    cfg = dict(type='RandomErasing', fill_std=0.5)
    random_erasing = build_from_cfg(cfg, PIPELINES)
    assert random_erasing.fill_std == [0.5, 0.5, 0.5]

    # test when erase_prob=0.
    results = construct_toy_data()
    cfg = dict(
        type='RandomErasing',
        erase_prob=0.,
        mode='const',
        fill_color=(255, 255, 255))
    random_erasing = build_from_cfg(cfg, PIPELINES)
    results = random_erasing(results)
    np.testing.assert_array_equal(results['img'], results['ori_img'])

    # test mode 'const'
    random.seed(0)
    np.random.seed(0)
    results = construct_toy_data()
    cfg = dict(
        type='RandomErasing',
        erase_prob=1.,
        mode='const',
        fill_color=(255, 255, 255))
    random_erasing = build_from_cfg(cfg, PIPELINES)
    results = random_erasing(results)

    expect_out = np.array([[1, 255, 3, 4], [5, 255, 7, 8], [9, 10, 11, 12]],
                          dtype=np.uint8)
    expect_out = np.stack([expect_out] * 3, axis=-1)
    np.testing.assert_array_equal(results['img'], expect_out)

    # test mode 'rand' with normal distribution
    random.seed(0)
    np.random.seed(0)
    results = construct_toy_data()
    cfg = dict(type='RandomErasing', erase_prob=1., mode='rand')
    random_erasing = build_from_cfg(cfg, PIPELINES)
    results = random_erasing(results)

    expect_out = results['ori_img']
    expect_out[:2, 1] = [[159, 98, 76], [14, 69, 122]]
    np.testing.assert_array_equal(results['img'], expect_out)

    # test mode 'rand' with uniform distribution
    random.seed(0)
    np.random.seed(0)
    results = construct_toy_data()
    cfg = dict(
        type='RandomErasing',
        erase_prob=1.,
        mode='rand',
        fill_std=(10, 255, 0))
    random_erasing = build_from_cfg(cfg, PIPELINES)
    results = random_erasing(results)

    expect_out = results['ori_img']
    expect_out[:2, 1] = [[113, 255, 128], [126, 83, 128]]
    np.testing.assert_array_equal(results['img'], expect_out)


def test_color_jitter():
    # read test image
    results = dict()
    img = mmcv.imread(
        osp.join(osp.dirname(__file__), '../../data/color.jpg'), 'color')
    original_img = copy.deepcopy(img)
    results['img'] = img
    results['img2'] = copy.deepcopy(img)
    results['img_shape'] = img.shape
    results['ori_shape'] = img.shape
    results['img_fields'] = ['img', 'img2']

    def reset_results(results, original_img):
        results['img'] = copy.deepcopy(original_img)
        results['img2'] = copy.deepcopy(original_img)
        results['img_shape'] = original_img.shape
        results['ori_shape'] = original_img.shape
        return results

    transform = dict(
        type='ColorJitter', brightness=0., contrast=0., saturation=0.)
    colorjitter_module = build_from_cfg(transform, PIPELINES)
    results = colorjitter_module(results)
    assert np.equal(results['img'], original_img).all()
    assert np.equal(results['img'], results['img2']).all()

    results = reset_results(results, original_img)
    transform = dict(
        type='ColorJitter', brightness=0.3, contrast=0.3, saturation=0.3)
    colorjitter_module = build_from_cfg(transform, PIPELINES)
    results = colorjitter_module(results)
    assert not np.equal(results['img'], original_img).all()


def test_lighting():
    # test assertion if eigval or eigvec is wrong type or length
    with pytest.raises(AssertionError):
        transform = dict(type='Lighting', eigval=1, eigvec=[[1, 0, 0]])
        build_from_cfg(transform, PIPELINES)
    with pytest.raises(AssertionError):
        transform = dict(type='Lighting', eigval=[1], eigvec=[1, 0, 0])
        build_from_cfg(transform, PIPELINES)
    with pytest.raises(AssertionError):
        transform = dict(
            type='Lighting', eigval=[1, 2], eigvec=[[1, 0, 0], [0, 1]])
        build_from_cfg(transform, PIPELINES)

    # read test image
    results = dict()
    img = mmcv.imread(
        osp.join(osp.dirname(__file__), '../../data/color.jpg'), 'color')
    original_img = copy.deepcopy(img)
    results['img'] = img
    results['img2'] = copy.deepcopy(img)
    results['img_shape'] = img.shape
    results['ori_shape'] = img.shape
    results['img_fields'] = ['img', 'img2']

    def reset_results(results, original_img):
        results['img'] = copy.deepcopy(original_img)
        results['img2'] = copy.deepcopy(original_img)
        results['img_shape'] = original_img.shape
        results['ori_shape'] = original_img.shape
        return results

    eigval = [0.2175, 0.0188, 0.0045]
    eigvec = [[-0.5675, 0.7192, 0.4009], [-0.5808, -0.0045, -0.8140],
              [-0.5836, -0.6948, 0.4203]]
    transform = dict(type='Lighting', eigval=eigval, eigvec=eigvec)
    lightening_module = build_from_cfg(transform, PIPELINES)
    results = lightening_module(results)
    assert not np.equal(results['img'], results['img2']).all()
    assert results['img'].dtype == float
    assert results['img2'].dtype == float

    results = reset_results(results, original_img)
    transform = dict(
        type='Lighting',
        eigval=eigval,
        eigvec=eigvec,
        alphastd=0.,
        to_rgb=False)
    lightening_module = build_from_cfg(transform, PIPELINES)
    results = lightening_module(results)
    assert np.equal(results['img'], original_img).all()
    assert np.equal(results['img'], results['img2']).all()
    assert results['img'].dtype == float
    assert results['img2'].dtype == float


def test_albu_transform():
    results = dict(
        img_prefix=osp.join(osp.dirname(__file__), '../../data'),
        img_info=dict(filename='color.jpg'))

    # Define simple pipeline
    load = dict(type='LoadImageFromFile')
    load = build_from_cfg(load, PIPELINES)

    albu_transform = dict(
        type='Albu', transforms=[dict(type='ChannelShuffle', p=1)])
    albu_transform = build_from_cfg(albu_transform, PIPELINES)

    normalize = dict(type='Normalize', mean=[0] * 3, std=[0] * 3, to_rgb=True)
    normalize = build_from_cfg(normalize, PIPELINES)

    # Execute transforms
    results = load(results)
    results = albu_transform(results)
    results = normalize(results)

    assert results['img'].dtype == np.float32
