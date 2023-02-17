# Copyright (c) OpenMMLab. All rights reserved.
import datetime
import sys
from unittest import TestCase

from mmengine import DefaultScope

from mmpretrain.utils import register_all_modules


class TestSetupEnv(TestCase):

    def test_register_all_modules(self):
        from mmpretrain.registry import DATASETS

        # not init default scope
        sys.modules.pop('mmpretrain.datasets', None)
        sys.modules.pop('mmpretrain.datasets.custom', None)
        DATASETS._module_dict.pop('CustomDataset', None)
        self.assertFalse('CustomDataset' in DATASETS.module_dict)
        register_all_modules(init_default_scope=False)
        self.assertTrue('CustomDataset' in DATASETS.module_dict)

        # init default scope
        sys.modules.pop('mmpretrain.datasets')
        sys.modules.pop('mmpretrain.datasets.custom')
        DATASETS._module_dict.pop('CustomDataset', None)
        self.assertFalse('CustomDataset' in DATASETS.module_dict)
        register_all_modules(init_default_scope=True)
        self.assertTrue('CustomDataset' in DATASETS.module_dict)
        self.assertEqual(DefaultScope.get_current_instance().scope_name,
                         'mmpretrain')

        # init default scope when another scope is init
        name = f'test-{datetime.datetime.now()}'
        DefaultScope.get_instance(name, scope_name='test')
        with self.assertWarnsRegex(
                Warning,
                'The current default scope "test" is not "mmpretrain"'):
            register_all_modules(init_default_scope=True)
