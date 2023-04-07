# Copyright (c) OpenMMLab. All rights reserved.
import importlib

from mmpretrain.registry import MODELS


@MODELS.register_module()
class AutoTokenizer:

    def __new__(cls,
                tokenizer_type: str = 'AutoTokenizer',
                name: str = 'bert-base-uncased',
                **kwargs):
        transformers = importlib.import_module('transformers')
        tokenizer = getattr(transformers, tokenizer_type)
        return tokenizer.from_pretrained(name, **kwargs)


@MODELS.register_module()
class BLIPTokenizer(AutoTokenizer):

    def __new__(cls, name: str, **kwargs):
        tokenizer = super().__new__(
            cls, tokenizer_type='BertTokenizer', name=name, **kwargs)

        tokenizer.add_special_tokens({'bos_token': '[DEC]'})
        tokenizer.add_special_tokens({'additional_special_tokens': ['[ENC]']})
        return tokenizer
