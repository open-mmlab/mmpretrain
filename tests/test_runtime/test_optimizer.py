# Copyright (c) OpenMMLab. All rights reserved.
import functools
from collections import OrderedDict
from copy import deepcopy
from typing import Iterable

import torch
import torch.nn as nn
from mmcv.runner import build_optimizer
from mmcv.runner.optimizer.builder import OPTIMIZERS
from mmcv.utils.registry import build_from_cfg
from torch.autograd import Variable
from torch.optim.optimizer import Optimizer

import mmcls.core  # noqa: F401

base_lr = 0.01
base_wd = 0.0001


def assert_equal(x, y):
    if isinstance(x, torch.Tensor) and isinstance(y, torch.Tensor):
        torch.testing.assert_allclose(x, y.to(x.device))
    elif isinstance(x, OrderedDict) and isinstance(y, OrderedDict):
        for x_value, y_value in zip(x.values(), y.values()):
            assert_equal(x_value, y_value)
    elif isinstance(x, dict) and isinstance(y, dict):
        assert x.keys() == y.keys()
        for key in x.keys():
            assert_equal(x[key], y[key])
    elif isinstance(x, str) and isinstance(y, str):
        assert x == y
    elif isinstance(x, Iterable) and isinstance(y, Iterable):
        assert len(x) == len(y)
        for x_item, y_item in zip(x, y):
            assert_equal(x_item, y_item)
    else:
        assert x == y


class SubModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(2, 2, kernel_size=1, groups=2)
        self.gn = nn.GroupNorm(2, 2)
        self.fc = nn.Linear(2, 2)
        self.param1 = nn.Parameter(torch.ones(1))

    def forward(self, x):
        return x


class ExampleModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.param1 = nn.Parameter(torch.ones(1))
        self.conv1 = nn.Conv2d(3, 4, kernel_size=1, bias=False)
        self.conv2 = nn.Conv2d(4, 2, kernel_size=1)
        self.bn = nn.BatchNorm2d(2)
        self.sub = SubModel()
        self.fc = nn.Linear(2, 1)

    def forward(self, x):
        return x


def check_lamb_optimizer(optimizer,
                         model,
                         bias_lr_mult=1,
                         bias_decay_mult=1,
                         norm_decay_mult=1,
                         dwconv_decay_mult=1):
    param_groups = optimizer.param_groups
    assert isinstance(optimizer, Optimizer)
    assert optimizer.defaults['lr'] == base_lr
    assert optimizer.defaults['weight_decay'] == base_wd
    model_parameters = list(model.parameters())
    assert len(param_groups) == len(model_parameters)
    for i, param in enumerate(model_parameters):
        param_group = param_groups[i]
        assert torch.equal(param_group['params'][0], param)
    # param1
    param1 = param_groups[0]
    assert param1['lr'] == base_lr
    assert param1['weight_decay'] == base_wd
    # conv1.weight
    conv1_weight = param_groups[1]
    assert conv1_weight['lr'] == base_lr
    assert conv1_weight['weight_decay'] == base_wd
    # conv2.weight
    conv2_weight = param_groups[2]
    assert conv2_weight['lr'] == base_lr
    assert conv2_weight['weight_decay'] == base_wd
    # conv2.bias
    conv2_bias = param_groups[3]
    assert conv2_bias['lr'] == base_lr * bias_lr_mult
    assert conv2_bias['weight_decay'] == base_wd * bias_decay_mult
    # bn.weight
    bn_weight = param_groups[4]
    assert bn_weight['lr'] == base_lr
    assert bn_weight['weight_decay'] == base_wd * norm_decay_mult
    # bn.bias
    bn_bias = param_groups[5]
    assert bn_bias['lr'] == base_lr
    assert bn_bias['weight_decay'] == base_wd * norm_decay_mult
    # sub.param1
    sub_param1 = param_groups[6]
    assert sub_param1['lr'] == base_lr
    assert sub_param1['weight_decay'] == base_wd
    # sub.conv1.weight
    sub_conv1_weight = param_groups[7]
    assert sub_conv1_weight['lr'] == base_lr
    assert sub_conv1_weight['weight_decay'] == base_wd * dwconv_decay_mult
    # sub.conv1.bias
    sub_conv1_bias = param_groups[8]
    assert sub_conv1_bias['lr'] == base_lr * bias_lr_mult
    assert sub_conv1_bias['weight_decay'] == base_wd * dwconv_decay_mult
    # sub.gn.weight
    sub_gn_weight = param_groups[9]
    assert sub_gn_weight['lr'] == base_lr
    assert sub_gn_weight['weight_decay'] == base_wd * norm_decay_mult
    # sub.gn.bias
    sub_gn_bias = param_groups[10]
    assert sub_gn_bias['lr'] == base_lr
    assert sub_gn_bias['weight_decay'] == base_wd * norm_decay_mult
    # sub.fc1.weight
    sub_fc_weight = param_groups[11]
    assert sub_fc_weight['lr'] == base_lr
    assert sub_fc_weight['weight_decay'] == base_wd
    # sub.fc1.bias
    sub_fc_bias = param_groups[12]
    assert sub_fc_bias['lr'] == base_lr * bias_lr_mult
    assert sub_fc_bias['weight_decay'] == base_wd * bias_decay_mult
    # fc1.weight
    fc_weight = param_groups[13]
    assert fc_weight['lr'] == base_lr
    assert fc_weight['weight_decay'] == base_wd
    # fc1.bias
    fc_bias = param_groups[14]
    assert fc_bias['lr'] == base_lr * bias_lr_mult
    assert fc_bias['weight_decay'] == base_wd * bias_decay_mult


def _test_state_dict(weight, bias, input, constructor):
    weight = Variable(weight, requires_grad=True)
    bias = Variable(bias, requires_grad=True)
    inputs = Variable(input)

    def fn_base(optimizer, weight, bias):
        optimizer.zero_grad()
        i = input_cuda if weight.is_cuda else inputs
        loss = (weight.mv(i) + bias).pow(2).sum()
        loss.backward()
        return loss

    optimizer = constructor(weight, bias)
    fn = functools.partial(fn_base, optimizer, weight, bias)

    # Prime the optimizer
    for _ in range(20):
        optimizer.step(fn)
    # Clone the weights and construct new optimizer for them
    weight_c = Variable(weight.data.clone(), requires_grad=True)
    bias_c = Variable(bias.data.clone(), requires_grad=True)
    optimizer_c = constructor(weight_c, bias_c)
    fn_c = functools.partial(fn_base, optimizer_c, weight_c, bias_c)
    # Load state dict
    state_dict = deepcopy(optimizer.state_dict())
    state_dict_c = deepcopy(optimizer.state_dict())
    optimizer_c.load_state_dict(state_dict_c)
    # Run both optimizations in parallel
    for _ in range(20):
        optimizer.step(fn)
        optimizer_c.step(fn_c)
        assert_equal(weight, weight_c)
        assert_equal(bias, bias_c)
    # Make sure state dict wasn't modified
    assert_equal(state_dict, state_dict_c)
    # Make sure state dict is deterministic with equal
    # but not identical parameters
    # NOTE: The state_dict of optimizers in PyTorch 1.5 have random keys,
    state_dict = deepcopy(optimizer.state_dict())
    state_dict_c = deepcopy(optimizer_c.state_dict())
    keys = state_dict['param_groups'][-1]['params']
    keys_c = state_dict_c['param_groups'][-1]['params']
    for key, key_c in zip(keys, keys_c):
        assert_equal(optimizer.state_dict()['state'][key],
                     optimizer_c.state_dict()['state'][key_c])
    # Make sure repeated parameters have identical representation in state dict
    optimizer_c.param_groups.extend(optimizer_c.param_groups)
    assert_equal(optimizer_c.state_dict()['param_groups'][0],
                 optimizer_c.state_dict()['param_groups'][1])

    # Check that state dict can be loaded even when we cast parameters
    # to a different type and move to a different device.
    if not torch.cuda.is_available():
        return

    input_cuda = Variable(inputs.data.float().cuda())
    weight_cuda = Variable(weight.data.float().cuda(), requires_grad=True)
    bias_cuda = Variable(bias.data.float().cuda(), requires_grad=True)
    optimizer_cuda = constructor(weight_cuda, bias_cuda)
    fn_cuda = functools.partial(fn_base, optimizer_cuda, weight_cuda,
                                bias_cuda)

    state_dict = deepcopy(optimizer.state_dict())
    state_dict_c = deepcopy(optimizer.state_dict())
    optimizer_cuda.load_state_dict(state_dict_c)

    # Make sure state dict wasn't modified
    assert_equal(state_dict, state_dict_c)

    for _ in range(20):
        optimizer.step(fn)
        optimizer_cuda.step(fn_cuda)
        assert_equal(weight, weight_cuda)
        assert_equal(bias, bias_cuda)

    # validate deepcopy() copies all public attributes
    def getPublicAttr(obj):
        return set(k for k in obj.__dict__ if not k.startswith('_'))

    assert_equal(getPublicAttr(optimizer), getPublicAttr(deepcopy(optimizer)))


def _test_basic_cases_template(weight, bias, inputs, constructor,
                               scheduler_constructors):
    """Copied from PyTorch."""
    weight = Variable(weight, requires_grad=True)
    bias = Variable(bias, requires_grad=True)
    inputs = Variable(inputs)
    optimizer = constructor(weight, bias)
    schedulers = []
    for scheduler_constructor in scheduler_constructors:
        schedulers.append(scheduler_constructor(optimizer))

    # to check if the optimizer can be printed as a string
    optimizer.__repr__()

    def fn():
        optimizer.zero_grad()
        y = weight.mv(inputs)
        if y.is_cuda and bias.is_cuda and y.get_device() != bias.get_device():
            y = y.cuda(bias.get_device())
        loss = (y + bias).pow(2).sum()
        loss.backward()
        return loss

    initial_value = fn().item()
    for _ in range(200):
        for scheduler in schedulers:
            scheduler.step()
        optimizer.step(fn)

    assert fn().item() < initial_value


def _test_basic_cases(constructor,
                      scheduler_constructors=None,
                      ignore_multidevice=False):
    """Copied from PyTorch."""
    if scheduler_constructors is None:
        scheduler_constructors = []
    _test_state_dict(
        torch.randn(10, 5), torch.randn(10), torch.randn(5), constructor)
    _test_basic_cases_template(
        torch.randn(10, 5), torch.randn(10), torch.randn(5), constructor,
        scheduler_constructors)
    # non-contiguous parameters
    _test_basic_cases_template(
        torch.randn(10, 5, 2)[..., 0],
        torch.randn(10, 2)[..., 0], torch.randn(5), constructor,
        scheduler_constructors)
    # CUDA
    if not torch.cuda.is_available():
        return
    _test_basic_cases_template(
        torch.randn(10, 5).cuda(),
        torch.randn(10).cuda(),
        torch.randn(5).cuda(), constructor, scheduler_constructors)
    # Multi-GPU
    if not torch.cuda.device_count() > 1 or ignore_multidevice:
        return
    _test_basic_cases_template(
        torch.randn(10, 5).cuda(0),
        torch.randn(10).cuda(1),
        torch.randn(5).cuda(0), constructor, scheduler_constructors)


def test_lamb_optimizer():
    model = ExampleModel()
    optimizer_cfg = dict(
        type='Lamb',
        lr=base_lr,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=base_wd,
        paramwise_cfg=dict(
            bias_lr_mult=2,
            bias_decay_mult=0.5,
            norm_decay_mult=0,
            dwconv_decay_mult=0.1))
    optimizer = build_optimizer(model, optimizer_cfg)
    check_lamb_optimizer(optimizer, model, **optimizer_cfg['paramwise_cfg'])

    _test_basic_cases(lambda weight, bias: build_from_cfg(
        dict(type='Lamb', params=[weight, bias], lr=base_lr), OPTIMIZERS))
