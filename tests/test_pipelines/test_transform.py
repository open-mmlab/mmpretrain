import copy
import os.path as osp

import mmcv
import numpy as np
import pytest
import torch
from mmcv.utils import build_from_cfg
from PIL import Image
from torchvision import transforms

from mmcls.datasets.builder import PIPELINES


def test_resize():
    # test assertion if size is smaller than 0
    with pytest.raises(AssertionError):
        transform = dict(type='Resize', size=-1)
        build_from_cfg(transform, PIPELINES)

    # test assertion if size is tuple but one value is smaller than 0
    with pytest.raises(AssertionError):
        transform = dict(type='Resize', size=(224, -1))
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

    # test repr
    transform = dict(type='Resize', size=224)
    resize_module = build_from_cfg(transform, PIPELINES)
    assert isinstance(repr(resize_module), str)

    # read test image
    results = dict()
    img = mmcv.imread(
        osp.join(osp.dirname(__file__), '../data/color.jpg'), 'color')
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

    # test resize when size is int
    transform = dict(type='Resize', size=224, interpolation='bilinear')
    resize_module = build_from_cfg(transform, PIPELINES)
    results = resize_module(results)
    assert np.equal(results['img'], results['img2']).all()
    assert results['img_shape'] == (224, 224, 3)

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

    # test repr
    transform = dict(type='CenterCrop', crop_size=224)
    center_crop_module = build_from_cfg(transform, PIPELINES)
    assert isinstance(repr(center_crop_module), str)

    # read test image
    results = dict()
    img = mmcv.imread(
        osp.join(osp.dirname(__file__), '../data/color.jpg'), 'color')
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
        osp.join(osp.dirname(__file__), '../data/color.jpg'), 'color')
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
