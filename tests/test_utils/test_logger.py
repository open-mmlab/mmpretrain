# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import tempfile

from mmcls.utils import get_root_logger, load_json_logs


def test_get_root_logger():
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


def test_load_json_logs():
    log_path = 'tests/data/test.log.json'
    log_dicts = load_json_logs([log_path])

    assert isinstance(log_dicts, list)
    assert len(log_dicts) == 1
    assert set(log_dicts[0].keys()) == set([1, 2, 3])
    assert set(log_dicts[0][1].keys()) == set(
        ['iter', 'lr', 'memory', 'data_time', 'time', 'mode'])
    assert isinstance(log_dicts[0][1]['lr'], list)
    assert len(log_dicts[0][1]['iter']) == 11
    assert len(log_dicts[0][1]['lr']) == 11
    assert len(log_dicts[0][2]['iter']) == 7
    assert len(log_dicts[0][2]['lr']) == 7
    assert log_dicts[0][3]['iter'] == [10, 20]
    assert log_dicts[0][3]['lr'] == [0.33305, 0.34759]
