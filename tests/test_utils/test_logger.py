# Copyright (c) OpenMMLab. All rights reserved.
import os
import os.path as osp
import tempfile

import mmcv.utils.logging

from mmcls.utils import get_root_logger, load_json_logs


def test_get_root_logger():
    # Reset the initialized log
    mmcv.utils.logging.logger_initialized = {}
    with tempfile.TemporaryDirectory() as tmpdirname:
        log_path = osp.join(tmpdirname, 'test.log')

        logger = get_root_logger(log_file=log_path)
        message1 = 'adhsuadghj'
        logger.info(message1)

        logger2 = get_root_logger()
        message2 = 'm,tkrgmkr'
        logger2.info(message2)

        with open(log_path, 'r') as f:
            lines = f.readlines()
            assert message1 in lines[0]
            assert message2 in lines[1]

        assert logger is logger2
        os.remove(log_path)


def test_load_json_logs():
    log_path = 'tests/data/test.logjson'
    log_dicts = load_json_logs([log_path])

    # test log_dicts
    assert isinstance(log_dicts, list)
    assert len(log_dicts) == 1

    # test log_dict
    log_dict = log_dicts[0]
    assert set(log_dict.keys()) == set([1, 2, 3])

    # test epoch dict in log_dict
    assert set(log_dict[1].keys()) == set(
        ['iter', 'lr', 'memory', 'data_time', 'time', 'mode'])
    assert isinstance(log_dict[1]['lr'], list)
    assert len(log_dict[1]['iter']) == 4
    assert len(log_dict[1]['lr']) == 4
    assert len(log_dict[2]['iter']) == 3
    assert len(log_dict[2]['lr']) == 3
    assert log_dict[3]['iter'] == [10, 20]
    assert log_dict[3]['lr'] == [0.33305, 0.34759]
