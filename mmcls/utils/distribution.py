# Copyright (c) OpenMMLab. All rights reserved.


def wrap_non_distributed_model(model, device='cuda', dim=0, *args, **kwargs):
    """Wrap module in non-distributed environment by device type.

    - For CUDA, wrap as :obj:`mmcv.parallel.MMDataParallel`.
    - For MPS, wrap as :obj:`mmcv.device.mps.MPSDataParallel`.
    - For CPU & IPU, not wrap the model.

    Args:
        model(:class:`nn.Module`): model to be parallelized.
        device(str): device type, cuda, cpu or mlu. Defaults to cuda.
        dim(int): Dimension used to scatter the data. Defaults to 0.

    Returns:
        model(nn.Module): the model to be parallelized.
    """
    if device == 'npu':
        from mmcv.device.npu import NPUDataParallel
        model = NPUDataParallel(model.npu(), dim=dim, *args, **kwargs)
    elif device == 'mlu':
        from mmcv.device.mlu import MLUDataParallel
        model = MLUDataParallel(model.mlu(), dim=dim, *args, **kwargs)
    elif device == 'cuda':
        from mmcv.parallel import MMDataParallel
        model = MMDataParallel(model.cuda(), dim=dim, *args, **kwargs)
    elif device == 'cpu':
        model = model.cpu()
    elif device == 'ipu':
        model = model.cpu()
    elif device == 'mps':
        from mmcv.device import mps
        model = mps.MPSDataParallel(model.to('mps'), dim=dim, *args, **kwargs)
    else:
        raise RuntimeError(f'Unavailable device "{device}"')

    return model


def wrap_distributed_model(model, device='cuda', *args, **kwargs):
    """Build DistributedDataParallel module by device type.

    - For CUDA, wrap as :obj:`mmcv.parallel.MMDistributedDataParallel`.
    - Other device types are not supported by now.

    Args:
        model(:class:`nn.Module`): module to be parallelized.
        device(str): device type, mlu or cuda.

    Returns:
        model(:class:`nn.Module`): the module to be parallelized

    References:
        .. [1] https://pytorch.org/docs/stable/generated/torch.nn.parallel.
               DistributedDataParallel.html
    """
    if device == 'npu':
        from mmcv.device.npu import NPUDistributedDataParallel
        from torch.npu import current_device
        model = NPUDistributedDataParallel(
            model.npu(), *args, device_ids=[current_device()], **kwargs)
    elif device == 'mlu':
        import os

        from mmcv.device.mlu import MLUDistributedDataParallel
        model = MLUDistributedDataParallel(
            model.mlu(),
            *args,
            device_ids=[int(os.environ['LOCAL_RANK'])],
            **kwargs)
    elif device == 'cuda':
        from mmcv.parallel import MMDistributedDataParallel
        from torch.cuda import current_device
        model = MMDistributedDataParallel(
            model.cuda(), *args, device_ids=[current_device()], **kwargs)
    else:
        raise RuntimeError(f'Unavailable device "{device}"')

    return model
