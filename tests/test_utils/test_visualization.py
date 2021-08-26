# Copyright (c) Open-MMLab. All rights reserved.
import os
import os.path as osp
import shutil
import tempfile
from unittest.mock import Mock, patch

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
    tmp_dir = osp.join(tempfile.gettempdir(), 'infos_image')
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

    # test show=True
    image = np.ones((10, 10, 3), np.uint8)
    result = {'pred_label': 1, 'pred_class': 'bird', 'pred_score': 0.98}

    def save_args(*args, **kwargs):
        args_list = ['args']
        args_list += [
            str(arg) for arg in args if isinstance(arg, (str, bool, int))
        ]
        args_list += [
            f'{k}-{v}' for k, v in kwargs.items()
            if isinstance(v, (str, bool, int))
        ]
        out_path = osp.join(tmp_dir, '_'.join(args_list))
        with open(out_path, 'w') as f:
            f.write('test')

    with patch('matplotlib.pyplot.show', save_args), \
            patch('matplotlib.pyplot.pause', save_args):
        vis.imshow_infos(image, result, show=True, wait_time=5)
        assert osp.exists(osp.join(tmp_dir, 'args_block-False'))
        assert osp.exists(osp.join(tmp_dir, 'args_5'))

        vis.imshow_infos(image, result, show=True, wait_time=0)
        assert osp.exists(osp.join(tmp_dir, 'args'))

    # test adaptive dpi
    def mock_fig_manager():
        fig_manager = Mock()
        fig_manager.window.winfo_screenheight = Mock(return_value=1440)
        return fig_manager

    with patch('matplotlib.pyplot.get_current_fig_manager',
               mock_fig_manager), patch('matplotlib.pyplot.show'):
        vis.imshow_infos(image, result, show=True)

    shutil.rmtree(tmp_dir)
