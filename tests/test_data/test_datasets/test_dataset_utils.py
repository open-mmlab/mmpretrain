# Copyright (c) OpenMMLab. All rights reserved.
import random
import string
import tempfile

from mmcls.datasets.utils import check_integrity, rm_suffix


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
    tmp_file = tempfile.NamedTemporaryFile()
    assert check_integrity(tmp_file.name, md5=None)
    assert not check_integrity(tmp_file.name, md5=2333)
