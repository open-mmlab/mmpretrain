# Copyright (c) OpenMMLab. All rights reserved.
import pytest
import torch

from mmcls.models.losses.utils import convert_to_one_hot


def ori_convert_to_one_hot(targets: torch.Tensor, classes) -> torch.Tensor:
    assert (torch.max(targets).item() <
            classes), 'Class Index must be less than number of classes'
    one_hot_targets = torch.zeros((targets.shape[0], classes),
                                  dtype=torch.long,
                                  device=targets.device)
    one_hot_targets.scatter_(1, targets.long(), 1)
    return one_hot_targets


def test_convert_to_one_hot():
    # label should smaller than classes
    targets = torch.tensor([1, 2, 3, 8, 5])
    classes = 5
    with pytest.raises(AssertionError):
        _ = convert_to_one_hot(targets, classes)

    # test with original impl
    classes = 10
    targets = torch.randint(high=classes, size=(10, 1))
    ori_one_hot_targets = torch.zeros((targets.shape[0], classes),
                                      dtype=torch.long,
                                      device=targets.device)
    ori_one_hot_targets.scatter_(1, targets.long(), 1)
    one_hot_targets = convert_to_one_hot(targets, classes)
    assert torch.equal(ori_one_hot_targets, one_hot_targets)


# test cuda version
@pytest.mark.skipif(
    not torch.cuda.is_available(), reason='requires CUDA support')
def test_convert_to_one_hot_cuda():
    # test with original impl
    classes = 10
    targets = torch.randint(high=classes, size=(10, 1)).cuda()
    ori_one_hot_targets = torch.zeros((targets.shape[0], classes),
                                      dtype=torch.long,
                                      device=targets.device)
    ori_one_hot_targets.scatter_(1, targets.long(), 1)
    one_hot_targets = convert_to_one_hot(targets, classes)
    assert torch.equal(ori_one_hot_targets, one_hot_targets)
    assert ori_one_hot_targets.device == one_hot_targets.device
