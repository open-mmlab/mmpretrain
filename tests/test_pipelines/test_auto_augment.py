import copy

import numpy as np
import pytest
from mmcv.utils import build_from_cfg

from mmcls.datasets.builder import PIPELINES


def construct_toy_data():
    img = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]],
                   dtype=np.uint8)
    img = np.stack([img, img, img], axis=-1)
    results = dict()
    # image
    results['ori_img'] = img
    results['img'] = img
    results['img2'] = copy.deepcopy(img)
    results['img_shape'] = img.shape
    results['ori_shape'] = img.shape
    results['img_fields'] = ['img', 'img2']
    return results


def test_shear():
    # test assertion for invalid type of magnitude
    with pytest.raises(AssertionError):
        transform = dict(type='Shear', magnitude=None)
        build_from_cfg(transform, PIPELINES)

    # test assertion for invalid pad_val
    with pytest.raises(AssertionError):
        transform = dict(type='Shear', magnitude=0.5, pad_val=(0, 0))
        build_from_cfg(transform, PIPELINES)

    # test assertion for invalid value of prob
    with pytest.raises(AssertionError):
        transform = dict(type='Shear', magnitude=0.5, prob=100)
        build_from_cfg(transform, PIPELINES)

    # test assertion for invalid direction
    with pytest.raises(AssertionError):
        transform = dict(type='Shear', magnitude=0.5, direction='diagonal')
        build_from_cfg(transform, PIPELINES)

    # test assertion for invalid value of random_negative_prob
    with pytest.raises(AssertionError):
        transform = dict(type='Shear', magnitude=0.5, random_negative_prob=100)
        build_from_cfg(transform, PIPELINES)

    # test case when magnitude = 0, therefore no shear
    results = construct_toy_data()
    transform = dict(type='Shear', magnitude=0., prob=1.)
    pipeline = build_from_cfg(transform, PIPELINES)
    results = pipeline(results)
    assert (results['img'] == results['ori_img']).all()

    # test case when prob = 0, therefore no shear
    results = construct_toy_data()
    transform = dict(type='Shear', magnitude=0.5, prob=0.)
    pipeline = build_from_cfg(transform, PIPELINES)
    results = pipeline(results)
    assert (results['img'] == results['ori_img']).all()

    # test shear horizontally, magnitude=1
    results = construct_toy_data()
    transform = dict(
        type='Shear', magnitude=1, pad_val=0, prob=1., random_negative_prob=0.)
    pipeline = build_from_cfg(transform, PIPELINES)
    results = pipeline(results)
    sheared_img = np.array([[1, 2, 3, 4], [0, 5, 6, 7], [0, 0, 9, 10]],
                           dtype=np.uint8)
    sheared_img = np.stack([sheared_img, sheared_img, sheared_img], axis=-1)
    assert (results['img'] == sheared_img).all()
    assert (results['img'] == results['img2']).all()

    # test shear vertically, magnitude=-1
    results = construct_toy_data()
    transform = dict(
        type='Shear',
        magnitude=-1,
        pad_val=0,
        prob=1.,
        direction='vertical',
        random_negative_prob=0.)
    pipeline = build_from_cfg(transform, PIPELINES)
    results = pipeline(results)
    sheared_img = np.array([[1, 6, 11, 0], [5, 10, 0, 0], [9, 0, 0, 0]],
                           dtype=np.uint8)
    sheared_img = np.stack([sheared_img, sheared_img, sheared_img], axis=-1)
    assert (results['img'] == sheared_img).all()

    # test shear vertically, magnitude=1, random_negative_prob=1
    results = construct_toy_data()
    transform = dict(
        type='Shear',
        magnitude=1,
        pad_val=0,
        prob=1.,
        direction='vertical',
        random_negative_prob=1.)
    pipeline = build_from_cfg(transform, PIPELINES)
    results = pipeline(results)
    sheared_img = np.array([[1, 6, 11, 0], [5, 10, 0, 0], [9, 0, 0, 0]],
                           dtype=np.uint8)
    sheared_img = np.stack([sheared_img, sheared_img, sheared_img], axis=-1)
    assert (results['img'] == sheared_img).all()


def test_translate():
    # test assertion for invalid type of magnitude
    with pytest.raises(AssertionError):
        transform = dict(type='Translate', magnitude=None)
        build_from_cfg(transform, PIPELINES)

    # test assertion for invalid pad_val
    with pytest.raises(AssertionError):
        transform = dict(type='Translate', magnitude=0.5, pad_val=(0, 0))
        build_from_cfg(transform, PIPELINES)

    # test assertion for invalid value of prob
    with pytest.raises(AssertionError):
        transform = dict(type='Translate', magnitude=0.5, prob=100)
        build_from_cfg(transform, PIPELINES)

    # test assertion for invalid direction
    with pytest.raises(AssertionError):
        transform = dict(type='Translate', magnitude=0.5, direction='diagonal')
        build_from_cfg(transform, PIPELINES)

    # test assertion for invalid value of random_negative_prob
    with pytest.raises(AssertionError):
        transform = dict(
            type='Translate', magnitude=0.5, random_negative_prob=100)
        build_from_cfg(transform, PIPELINES)

    # test case when magnitude=0, therefore no translate
    results = construct_toy_data()
    transform = dict(type='Translate', magnitude=0., prob=1.)
    pipeline = build_from_cfg(transform, PIPELINES)
    results = pipeline(results)
    assert (results['img'] == results['ori_img']).all()

    # test case when prob=0, therefore no shear
    results = construct_toy_data()
    transform = dict(type='Translate', magnitude=1., prob=0.)
    pipeline = build_from_cfg(transform, PIPELINES)
    results = pipeline(results)
    assert (results['img'] == results['ori_img']).all()

    # test translate horizontally, magnitude=0.5
    results = construct_toy_data()
    transform = dict(
        type='Translate',
        magnitude=0.5,
        pad_val=0,
        prob=1.,
        random_negative_prob=0.)
    pipeline = build_from_cfg(transform, PIPELINES)
    results = pipeline(results)
    translated_img = np.array([[0, 0, 1, 2], [0, 0, 5, 6], [0, 0, 9, 10]],
                              dtype=np.uint8)
    translated_img = np.stack([translated_img, translated_img, translated_img],
                              axis=-1)
    assert (results['img'] == translated_img).all()
    assert (results['img'] == results['img2']).all()

    # test translate vertically, magnitude=-0.5
    results = construct_toy_data()
    transform = dict(
        type='Translate',
        magnitude=-0.5,
        pad_val=0,
        prob=1.,
        direction='vertical',
        random_negative_prob=0.)
    pipeline = build_from_cfg(transform, PIPELINES)
    results = pipeline(results)
    translated_img = np.array([[9, 10, 11, 12], [0, 0, 0, 0], [0, 0, 0, 0]],
                              dtype=np.uint8)
    translated_img = np.stack([translated_img, translated_img, translated_img],
                              axis=-1)
    assert (results['img'] == translated_img).all()

    # test translate vertically, magnitude=0.5, random_negative_prob=1
    results = construct_toy_data()
    transform = dict(
        type='Translate',
        magnitude=0.5,
        pad_val=0,
        prob=1.,
        direction='vertical',
        random_negative_prob=1.)
    pipeline = build_from_cfg(transform, PIPELINES)
    results = pipeline(results)
    translated_img = np.array([[9, 10, 11, 12], [0, 0, 0, 0], [0, 0, 0, 0]],
                              dtype=np.uint8)
    translated_img = np.stack([translated_img, translated_img, translated_img],
                              axis=-1)
    assert (results['img'] == translated_img).all()


def test_invert():
    # test assertion for invalid value of prob
    with pytest.raises(AssertionError):
        transform = dict(type='Invert', prob=100)
        build_from_cfg(transform, PIPELINES)

    # test case when prob=0, therefore no invert
    results = construct_toy_data()
    transform = dict(type='Invert', prob=0.)
    pipeline = build_from_cfg(transform, PIPELINES)
    results = pipeline(results)
    assert (results['img'] == results['ori_img']).all()

    # test case when prob=1
    results = construct_toy_data()
    transform = dict(type='Invert', prob=1.)
    pipeline = build_from_cfg(transform, PIPELINES)
    results = pipeline(results)
    inverted_img = np.array(
        [[254, 253, 252, 251], [250, 249, 248, 247], [246, 245, 244, 243]],
        dtype=np.uint8)
    inverted_img = np.stack([inverted_img, inverted_img, inverted_img],
                            axis=-1)
    assert (results['img'] == inverted_img).all()
    assert (results['img'] == results['img2']).all()
