# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
from mmcv.transforms import Resize

from mmpretrain.datasets import GaussianBlur, MultiView, Solarize


def test_multi_view():
    original_img = np.ones((4, 4, 3), dtype=np.uint8)

    # test 1 pipeline with 2 views
    pipeline1 = [
        Resize(2),
        GaussianBlur(magnitude_range=(0.1, 2), magnitude_std='inf')
    ]

    transform = MultiView([pipeline1], 2)
    results = dict(img=original_img)
    results = transform(results)
    assert len(results['img']) == 2
    assert results['img'][0].shape == (2, 2, 3)

    transform = MultiView([pipeline1], [2])
    results = dict(img=original_img)
    results = transform(results)
    assert len(results['img']) == 2
    assert results['img'][0].shape == (2, 2, 3)

    # test 2 pipeline with 3 views
    pipeline2 = [
        Solarize(thr=128),
        GaussianBlur(magnitude_range=(0.1, 2), magnitude_std='inf')
    ]
    transform = MultiView([pipeline1, pipeline2], [1, 2])

    results = dict(img=original_img)
    results = transform(results)
    assert len(results['img']) == 3
    assert results['img'][0].shape == (2, 2, 3)
    assert results['img'][1].shape == (4, 4, 3)

    # test repr
    assert isinstance(str(transform), str)
