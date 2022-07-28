# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase
from unittest.mock import patch

import mmcv

from mmcls.utils import auto_select_device


class TestAutoSelectDevice(TestCase):

    @patch.object(mmcv, '__version__', '1.6.0')
    @patch('mmcv.device.get_device', create=True)
    def test_mmcv(self, mock):
        auto_select_device()
        mock.assert_called_once()

    @patch.object(mmcv, '__version__', '1.5.0')
    @patch('torch.cuda.is_available', return_value=True)
    def test_cuda(self, mock):
        device = auto_select_device()
        self.assertEqual(device, 'cuda')

    @patch.object(mmcv, '__version__', '1.5.0')
    @patch('torch.cuda.is_available', return_value=False)
    def test_cpu(self, mock):
        device = auto_select_device()
        self.assertEqual(device, 'cpu')
