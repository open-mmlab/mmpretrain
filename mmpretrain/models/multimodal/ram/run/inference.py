# Copyright (c) OpenMMLab. All rights reserved.
import torch


def inference_ram(sample, model):

    with torch.no_grad():
        result = model.test_step(sample)

    return result


def inference_ram_openset(sample, model):
    with torch.no_grad():
        result = model.test_step(sample)

    return result


def inference(sample, model, transforms, mode='normal'):
    sample = transforms(sample)
    if sample['inputs'].ndim == 3:
        sample['inputs'] = sample['inputs'].unsqueeze(dim=0)
    assert mode in ['normal', 'openset'
                    ], 'mode of inference must be "normal" or "openset"'
    if mode == 'normal':
        return inference_ram(sample, model)
    else:
        return inference_ram_openset(sample, model)
