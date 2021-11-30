# Copyright (c) Open-MMLab. All rights reserved.
import os
import os.path as osp
import tempfile
from unittest.mock import MagicMock

import matplotlib.pyplot as plt
import mmcv
import numpy as np
import pytest

from mmcls.core import visualization as vis


def test_color():
    assert vis.color_val_matplotlib(mmcv.Color.blue) == (0., 0., 1.)
    assert vis.color_val_matplotlib('green') == (0., 1., 0.)
    assert vis.color_val_matplotlib((1, 2, 3)) == (3 / 255, 2 / 255, 1 / 255)
    assert vis.color_val_matplotlib(100) == (100 / 255, 100 / 255, 100 / 255)
    assert vis.color_val_matplotlib(np.zeros(3, dtype=int)) == (0., 0., 0.)
    # forbid white color
    with pytest.raises(TypeError):
        vis.color_val_matplotlib([255, 255, 255])
    # forbid float
    with pytest.raises(TypeError):
        vis.color_val_matplotlib(1.0)
    # overflowed
    with pytest.raises(AssertionError):
        vis.color_val_matplotlib((0, 0, 500))


def test_imshow_infos():
    tmp_dir = osp.join(tempfile.gettempdir(), 'image_infos')
    tmp_filename = osp.join(tmp_dir, 'image.jpg')

    image = np.ones((10, 10, 3), np.uint8)
    result = {'pred_label': 1, 'pred_class': 'bird', 'pred_score': 0.98}
    out_image = vis.imshow_infos(
        image, result, out_file=tmp_filename, show=False)
    assert osp.isfile(tmp_filename)
    assert image.shape == out_image.shape
    assert not np.allclose(image, out_image)
    os.remove(tmp_filename)

    # test grayscale images
    image = np.ones((10, 10), np.uint8)
    result = {'pred_label': 1, 'pred_class': 'bird', 'pred_score': 0.98}
    out_image = vis.imshow_infos(
        image, result, out_file=tmp_filename, show=False)
    assert osp.isfile(tmp_filename)
    assert image.shape == out_image.shape[:2]
    os.remove(tmp_filename)


def test_figure_context_manager():
    # test show multiple images with the same figure.
    images = [
        np.random.randint(0, 255, (100, 100, 3), np.uint8) for _ in range(5)
    ]
    result = {'pred_label': 1, 'pred_class': 'bird', 'pred_score': 0.98}

    with vis.ImshowInfosContextManager() as manager:
        fig_show = manager.fig_show
        fig_save = manager.fig_save

        # Test time out
        fig_show.canvas.start_event_loop = MagicMock()
        fig_show.canvas.end_event_loop = MagicMock()
        for image in images:
            ret, out_image = manager.put_img_infos(image, result, show=True)
            assert ret == 0
            assert image.shape == out_image.shape
            assert not np.allclose(image, out_image)
            assert fig_show is manager.fig_show
            assert fig_save is manager.fig_save

        # Test continue key
        fig_show.canvas.start_event_loop = (
            lambda _: fig_show.canvas.key_press_event(' '))
        for image in images:
            ret, out_image = manager.put_img_infos(image, result, show=True)
            assert ret == 0
            assert image.shape == out_image.shape
            assert not np.allclose(image, out_image)
            assert fig_show is manager.fig_show
            assert fig_save is manager.fig_save

        # Test close figure manually
        fig_show = manager.fig_show

        def destroy(*_, **__):
            fig_show.canvas.close_event()
            plt.close(fig_show)

        fig_show.canvas.start_event_loop = destroy
        ret, out_image = manager.put_img_infos(images[0], result, show=True)
        assert ret == 1
        assert image.shape == out_image.shape
        assert not np.allclose(image, out_image)
        assert fig_save is manager.fig_save
