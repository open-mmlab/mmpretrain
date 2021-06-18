import pytest
import torch

from mmcls.models.utils import PatchMerging


def cal_unfold_dim(dim, kernel_size, stride, padding=0, dilation=1):
    return (dim + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1


def test_patch_merging():
    settings = dict(
        input_resolution=(56, 56), in_channels=16, expansion_ratio=2)
    downsample = PatchMerging(**settings)

    # test forward with wrong dims
    with pytest.raises(AssertionError):
        inputs = torch.rand((1, 16, 56 * 56))
        downsample(inputs)

    # test patch merging forward
    inputs = torch.rand((1, 56 * 56, 16))
    out = downsample(inputs)
    assert downsample.output_resolution == (28, 28)
    assert out.shape == (1, 28 * 28, 32)

    # test different kernel_size in each direction
    downsample = PatchMerging(kernel_size=(2, 3), **settings)
    out = downsample(inputs)
    expected_dim = cal_unfold_dim(56, 2, 2) * cal_unfold_dim(56, 3, 3)
    assert downsample.sampler.kernel_size == (2, 3)
    assert downsample.output_resolution == (cal_unfold_dim(56, 2, 2),
                                            cal_unfold_dim(56, 3, 3))
    assert out.shape == (1, expected_dim, 32)

    # test default stride
    downsample = PatchMerging(kernel_size=6, **settings)
    assert downsample.sampler.stride == (6, 6)

    # test stride=3
    downsample = PatchMerging(kernel_size=6, stride=3, **settings)
    out = downsample(inputs)
    assert downsample.sampler.stride == (3, 3)
    assert out.shape == (1, cal_unfold_dim(56, 6, stride=3)**2, 32)

    # test padding
    downsample = PatchMerging(kernel_size=6, padding=2, **settings)
    out = downsample(inputs)
    assert downsample.sampler.padding == (2, 2)
    assert out.shape == (1, cal_unfold_dim(56, 6, 6, padding=2)**2, 32)

    # test dilation
    downsample = PatchMerging(kernel_size=6, dilation=2, **settings)
    out = downsample(inputs)
    assert downsample.sampler.dilation == (2, 2)
    assert out.shape == (1, cal_unfold_dim(56, 6, 6, dilation=2)**2, 32)
