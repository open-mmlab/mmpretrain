# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase
from unittest.mock import MagicMock, patch

from mmpretrain.engine import ClassNumCheckHook


class TestClassNumCheckHook(TestCase):

    def setUp(self):
        self.runner = MagicMock()
        self.dataset = MagicMock()
        self.hook = ClassNumCheckHook()

    def test_check_head(self):
        # check sequence of string
        with self.assertRaises(AssertionError):
            self.hook._check_head(self.runner, self.dataset)

        # check no CLASSES
        with patch.object(self.runner.logger, 'warning') as mock:
            self.dataset.CLASSES = None
            self.hook._check_head(self.runner, self.dataset)
            mock.assert_called_once()

        # check no modules
        self.dataset.CLASSES = ['str'] * 10
        self.hook._check_head(self.runner, self.dataset)

        # check number of classes not match
        self.dataset.CLASSES = ['str'] * 10
        module1 = MagicMock(spec_set=True)
        module2 = MagicMock(num_classes=5)
        self.runner.model.named_modules.return_value = iter([(None, module1),
                                                             (None, module2)])
        with self.assertRaises(AssertionError):
            self.hook._check_head(self.runner, self.dataset)

    def test_before_train(self):
        with patch.object(self.hook, '_check_head') as mock:
            self.hook.before_train(self.runner)
            mock.assert_called_once()

    def test_before_val(self):
        with patch.object(self.hook, '_check_head') as mock:
            self.hook.before_val(self.runner)
            mock.assert_called_once()

    def test_before_test(self):
        with patch.object(self.hook, '_check_head') as mock:
            self.hook.before_test(self.runner)
            mock.assert_called_once()
