# Copyright (c) OpenMMLab. All rights reserved.
import os
import os.path as osp

from mmengine.config import Config


def get_config_list(path, file_list):
    files = os.listdir(path)
    for f in files:
        tmp_path = osp.join(path, f)
        if osp.isdir(tmp_path):
            get_config_list(tmp_path, file_list)
        elif f[-2:] == 'py':
            file_list.append(tmp_path)


def test_configs():
    path = osp.join(osp.dirname(__file__), '..', 'configs')
    config_list = []
    get_config_list(path, config_list)

    for config in config_list:
        cfg = Config.fromfile(config)
        assert isinstance(cfg, Config)
