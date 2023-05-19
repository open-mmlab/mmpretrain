# Copyright (c) OpenMMLab. All rights reserved.
import os

from transformers import (AutoTokenizer, BartTokenizer, BertTokenizer,
                          BertTokenizerFast, LlamaTokenizer)

from .huggingface import register_hf_tokenizer

register_hf_tokenizer(AutoTokenizer)
register_hf_tokenizer(LlamaTokenizer)


@register_hf_tokenizer()
class BlipTokenizer(BertTokenizerFast):
    """"BlipTokenizer inherit BertTokenizerFast (fast, Rust-based)."""

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path,
        *init_inputs,
        **kwargs,
    ):
        os.environ['TOKENIZERS_PARALLELISM'] = 'true'

        tokenizer = super().from_pretrained(
            pretrained_model_name_or_path,
            *init_inputs,
            **kwargs,
        )
        tokenizer.add_special_tokens({'bos_token': '[DEC]'})
        tokenizer.add_special_tokens({'additional_special_tokens': ['[ENC]']})
        return tokenizer


@register_hf_tokenizer()
class Blip2Tokenizer(BertTokenizer):

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path,
        *init_inputs,
        **kwargs,
    ):
        tokenizer = super().from_pretrained(
            pretrained_model_name_or_path,
            *init_inputs,
            **kwargs,
        )
        tokenizer.add_special_tokens({'bos_token': '[DEC]'})
        return tokenizer


@register_hf_tokenizer()
class OFATokenizer(BartTokenizer):

    vocab_files_names = {
        'vocab_file': 'vocab.json',
        'merges_file': 'merges.txt'
    }

    pretrained_vocab_files_map = {
        'vocab_file': {
            'OFA-Sys/OFA-tiny':
            'https://huggingface.co/OFA-Sys/OFA-tiny/blob/main/vocab.json',
            'OFA-Sys/OFA-medium':
            'https://huggingface.co/OFA-Sys/OFA-medium/blob/main/vocab.json',
            'OFA-Sys/OFA-base':
            'https://huggingface.co/OFA-Sys/OFA-base/blob/main/vocab.json',
            'OFA-Sys/OFA-large':
            'https://huggingface.co/OFA-Sys/OFA-large/blob/main/vocab.json',
        },
        'merges_file': {
            'OFA-Sys/OFA-tiny':
            'https://huggingface.co/OFA-Sys/OFA-tiny/blob/main/merges.txt',
            'OFA-Sys/OFA-medium':
            'https://huggingface.co/OFA-Sys/OFA-medium/blob/main/merges.txt',
            'OFA-Sys/OFA-base':
            'https://huggingface.co/OFA-Sys/OFA-base/blob/main/merges.txt',
            'OFA-Sys/OFA-large':
            'https://huggingface.co/OFA-Sys/OFA-large/blob/main/merges.txt',
        },
    }

    max_model_input_sizes = {
        'OFA-Sys/OFA-tiny': 1024,
        'OFA-Sys/OFA-medium': 1024,
        'OFA-Sys/OFA-base': 1024,
        'OFA-Sys/OFA-large': 1024,
    }

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path,
        *init_inputs,
        **kwargs,
    ):
        num_bins = kwargs.pop('num_bins', 1000)
        tokenizer = super().from_pretrained(
            pretrained_model_name_or_path,
            *init_inputs,
            **kwargs,
        )
        length = len(tokenizer)
        tokenizer.add_tokens(['<code_{}>'.format(i) for i in range(8192)])
        tokenizer.code_offset = length
        tokenizer.add_tokens(['<bin_{}>'.format(i) for i in range(num_bins)])
        tokenizer.bin_offset = length + 8192
        tokenizer.num_bins = num_bins
        return tokenizer
