# Copyright (c) OpenMMLab. All rights reserved.
import re

from mmpretrain.models.peft import LoRAModel


def test_lora_model():
    module = dict(
        type='VisionTransformer',
        arch='base',
        img_size=224,
        patch_size=16,
        drop_path_rate=0.1,
        out_type='avg_featmap',
        final_norm=False)

    lora_cfg = dict(
        module=module,
        alpha=1,
        rank=4,
        drop_rate=0.1,
        targets=[
            dict(type='qkv'),
            dict(type='.*proj', alpha=2, rank=2, drop_rate=0.2),
        ])

    lora_model = LoRAModel(**lora_cfg)

    # test replace module
    for name, module in lora_model.named_modules():
        if name.endswith('qkv'):
            assert module.scaling == 0.25
        if re.fullmatch('.*proj', name):
            assert module.scaling == 1

    # test freeze module
    for name, param in lora_model.named_parameters():
        if 'lora_' in name:
            assert param.requires_grad
        else:
            assert not param.requires_grad

    # test get state dict
    state_dict = lora_model.state_dict()
    assert len(state_dict) != 0
    for name, param in state_dict.items():
        assert 'lora_' in name

    # test load state dict
    incompatible_keys = lora_model.load_state_dict(state_dict, strict=True)
    assert str(incompatible_keys) == '<All keys matched successfully>'
