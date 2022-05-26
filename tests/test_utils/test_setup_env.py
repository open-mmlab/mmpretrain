# Copyright (c) OpenMMLab. All rights reserved.
import datetime
import multiprocessing as mp
import os
import platform
import sys
from unittest import TestCase

import cv2
from mmcv import Config
from mmengine import DefaultScope

from mmcls.utils import register_all_modules, setup_multi_processes


class TestSetupEnv(TestCase):

    def test_register_all_modules(self):
        from mmcls.registry import DATASETS

        # not init default scope
        sys.modules.pop('mmcls.datasets', None)
        sys.modules.pop('mmcls.datasets.custom', None)
        DATASETS._module_dict.pop('CustomDataset', None)
        self.assertFalse('CustomDataset' in DATASETS.module_dict)
        register_all_modules(init_default_scope=False)
        self.assertTrue('CustomDataset' in DATASETS.module_dict)

        # init default scope
        sys.modules.pop('mmcls.datasets')
        sys.modules.pop('mmcls.datasets.custom')
        DATASETS._module_dict.pop('CustomDataset', None)
        self.assertFalse('CustomDataset' in DATASETS.module_dict)
        register_all_modules(init_default_scope=True)
        self.assertTrue('CustomDataset' in DATASETS.module_dict)
        self.assertEqual(DefaultScope.get_current_instance().scope_name,
                         'mmcls')

        # init default scope when another scope is init
        name = f'test-{datetime.datetime.now()}'
        DefaultScope.get_instance(name, scope_name='test')
        with self.assertWarnsRegex(
                Warning, 'The current default scope "test" is not "mmcls"'):
            register_all_modules(init_default_scope=True)


def test_setup_multi_processes():
    # temp save system setting
    sys_start_mehod = mp.get_start_method(allow_none=True)
    sys_cv_threads = cv2.getNumThreads()
    # pop and temp save system env vars
    sys_omp_threads = os.environ.pop('OMP_NUM_THREADS', default=None)
    sys_mkl_threads = os.environ.pop('MKL_NUM_THREADS', default=None)

    # test config without setting env
    config = dict(data=dict(workers_per_gpu=2))
    cfg = Config(config)
    setup_multi_processes(cfg)
    assert os.getenv('OMP_NUM_THREADS') == '1'
    assert os.getenv('MKL_NUM_THREADS') == '1'
    # when set to 0, the num threads will be 1
    assert cv2.getNumThreads() == 1
    if platform.system() != 'Windows':
        assert mp.get_start_method() == 'fork'

    # test num workers <= 1
    os.environ.pop('OMP_NUM_THREADS')
    os.environ.pop('MKL_NUM_THREADS')
    config = dict(data=dict(workers_per_gpu=0))
    cfg = Config(config)
    setup_multi_processes(cfg)
    assert 'OMP_NUM_THREADS' not in os.environ
    assert 'MKL_NUM_THREADS' not in os.environ

    # test manually set env var
    os.environ['OMP_NUM_THREADS'] = '4'
    config = dict(data=dict(workers_per_gpu=2))
    cfg = Config(config)
    setup_multi_processes(cfg)
    assert os.getenv('OMP_NUM_THREADS') == '4'

    # test manually set opencv threads and mp start method
    config = dict(
        data=dict(workers_per_gpu=2),
        opencv_num_threads=4,
        mp_start_method='spawn')
    cfg = Config(config)
    setup_multi_processes(cfg)
    assert cv2.getNumThreads() == 4
    assert mp.get_start_method() == 'spawn'

    # revert setting to avoid affecting other programs
    if sys_start_mehod:
        mp.set_start_method(sys_start_mehod, force=True)
    cv2.setNumThreads(sys_cv_threads)
    if sys_omp_threads:
        os.environ['OMP_NUM_THREADS'] = sys_omp_threads
    else:
        os.environ.pop('OMP_NUM_THREADS')
    if sys_mkl_threads:
        os.environ['MKL_NUM_THREADS'] = sys_mkl_threads
    else:
        os.environ.pop('MKL_NUM_THREADS')
