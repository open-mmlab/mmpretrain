import pytest
import torch

from mmcls.core import mAP


def test_mAP():
    target = torch.Tensor([[1, 1, -1, 0], [-1, 1, -1, 0], [-1, 0, 1, 0],
                           [1, -1, -1, 0]])
    pred = torch.Tensor([[0.9, 0.8, 0.3, 0.2], [0.1, 0.2, 0.2, 0.1],
                         [0.7, 0.5, 0.9, 0.3], [0.8, 0.1, 0.1, 0.2]])

    # target and pred should both be np.ndarray or torch.Tensor
    with pytest.raises(TypeError):
        target_list = target.tolist()
        _ = mAP(pred, target_list)

    # target and pred should be in the same shape
    with pytest.raises(AssertionError):
        target_shorter = target[:-1]
        _ = mAP(pred, target_shorter)

    assert mAP(pred, target) == pytest.approx(74.99999)
