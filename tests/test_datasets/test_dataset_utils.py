# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import random
import string
from unittest.mock import patch

import pytest

from mmpretrain.datasets.utils import (check_integrity,
                                       open_maybe_compressed_file, rm_suffix)


def test_dataset_utils():
    # test rm_suffix
    assert rm_suffix('a.jpg') == 'a'
    assert rm_suffix('a.bak.jpg') == 'a.bak'
    assert rm_suffix('a.bak.jpg', suffix='.jpg') == 'a.bak'
    assert rm_suffix('a.bak.jpg', suffix='.bak.jpg') == 'a'

    # test check_integrity
    rand_file = ''.join(random.sample(string.ascii_letters, 10))
    assert not check_integrity(rand_file, md5=None)
    assert not check_integrity(rand_file, md5=2333)
    test_file = osp.join(osp.dirname(__file__), '../data/color.jpg')
    assert check_integrity(test_file, md5='08252e5100cb321fe74e0e12a724ce14')
    assert not check_integrity(test_file, md5=2333)


@pytest.mark.parametrize('method,path', [('gzip.open', 'abc.gz'),
                                         ('lzma.open', 'abc.xz'),
                                         ('builtins.open', 'abc.txt'),
                                         (None, 1)])
def test_open_maybe_compressed_file(method, path):
    if method:
        with patch(method) as mock:
            open_maybe_compressed_file(path)
            mock.assert_called()
    else:
        assert open_maybe_compressed_file(path) == path
