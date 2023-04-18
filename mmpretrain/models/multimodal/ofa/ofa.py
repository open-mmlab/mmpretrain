# Copyright (c) OpenMMLab. All rights reserved.
import string
from collections import defaultdict
from functools import partial
from typing import Optional, Union

import mmengine
import torch
from mmengine.model import BaseModel

from mmpretrain.datasets import CleanCaption
from mmpretrain.registry import MODELS, TOKENIZER
from mmpretrain.structures import DataSample
from .ofa_modules import OFAEncoderDecoder


class TreeNode():

    def __init__(self):
        self.child = defaultdict(TreeNode)


class Trie:

    def __init__(self, eos):
        self.root = TreeNode()
        self.eos = eos

    def insert(self, word):
        cur = self.root
        for c in word:
            cur = cur.child[c]

    def get_next_layer(self, word):
        cur = self.root
        for c in word:
            cur = cur.child.get(c)
            if cur is None:
                return [self.eos]
        return list(cur.child.keys())


def apply_constraint(
    input_ids: torch.Tensor,
    logits: torch.Tensor,
    decoder_prompts: Optional[list],
    num_beams: int,
    constraint_trie: Trie = None,
):
    if decoder_prompts is None and constraint_trie is None:
        return logits

    mask = logits.new_zeros(logits[:, -1, :].size(), dtype=torch.bool)
    input_ids = input_ids.view(-1, num_beams, input_ids.shape[-1])
    for batch_id, beam_sent in enumerate(input_ids):
        for beam_id, sent in enumerate(beam_sent):
            if decoder_prompts is None:
                prompt_len = 0
            else:
                prompt_len = len(decoder_prompts[batch_id])

            if sent.size(0) - 1 < prompt_len:
                allowed_tokens = [decoder_prompts[batch_id][sent.size(0) - 1]]
                mask[batch_id * num_beams + beam_id, allowed_tokens] = True
            elif constraint_trie is not None:
                answer_tokens = [0] + sent[prompt_len + 1:].tolist()
                allowed_tokens = constraint_trie.get_next_layer(answer_tokens)
                mask[batch_id * num_beams + beam_id, allowed_tokens] = True
            else:
                mask[batch_id * num_beams + beam_id, :] = True
    logits[:, -1, :].masked_fill_(~mask, float('-inf'))
    return logits


@MODELS.register_module()
class OFA(BaseModel):
    """The OFA model for multiple tasks.

    Args:
        encoder_cfg (dict): The config of the encoder, accept the keyword
            arguments of [`~OFAEncoder`].
        decoder_cfg (dict): The config of the decoder, accept the keyword
            arguments of [`~OFADecoder`].
        padding_idx (int): The index of the padding token.
        vocab_size (int): The size of the vocabulary.
        embedding_dim (int): The embedding dimensions of both the encoder
            and the decoder.
        tokenizer (dict | PreTrainedTokenizer): The tokenizer to encode
            the text.
        task (str):
        prompt (str, optional):
        ans2label (str | dict | None):
        generation_cfg (dict): The extra generation config, accept the keyword
            arguments of [~`transformers.GenerationConfig`].
            Defaults to an empty dict.
        init_cfg (dict, optional): The initialization config. Defaults to None.
    """
    support_tasks = {'caption', 'vqa'}

    def __init__(
        self,
        encoder_cfg,
        decoder_cfg,
        vocab_size,
        embedding_dim,
        tokenizer,
        task='caption',
        prompt=None,
        ans2label: Union[dict, str, None] = None,
        generation_cfg=dict(),
        data_preprocessor: Optional[dict] = None,
        init_cfg=None,
    ):
        if data_preprocessor is None:
            data_preprocessor = {}
        data_preprocessor.setdefault('type', 'MultiModalDataPreprocessor')
        data_preprocessor = MODELS.build(data_preprocessor)

        super().__init__(
            init_cfg=init_cfg, data_preprocessor=data_preprocessor)

        if isinstance(tokenizer, dict):
            self.tokenizer = TOKENIZER.build(tokenizer)
        else:
            self.tokenizer = tokenizer

        if task not in self.support_tasks:
            raise ValueError(f'Unsupported task {task}, please select '
                             f'the task from {self.support_tasks}.')

        self.prompt = prompt
        self.task = task

        if isinstance(ans2label, str):
            self.ans2label = mmengine.load(ans2label)
        else:
            self.ans2label = ans2label

        if self.task == 'vqa' and self.ans2label is not None:
            self.constraint_trie = Trie(eos=self.tokenizer.eos_token_id)
            answers = [f' {answer}' for answer in self.ans2label]
            answer_tokens = self.tokenizer(answers, padding=False)
            for answer_token in answer_tokens['input_ids']:
                self.constraint_trie.insert(answer_token)
        else:
            self.constraint_trie = None

        generation_cfg = {
            'num_beams': 5,
            'max_new_tokens': 20,
            'no_repeat_ngram_size': 3,
            **generation_cfg,
        }
        self.model = OFAEncoderDecoder(
            encoder_cfg=encoder_cfg,
            decoder_cfg=decoder_cfg,
            padding_idx=self.tokenizer.pad_token_id,
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            generation_cfg=generation_cfg,
        )

    def forward(
        self,
        inputs: dict,
        data_samples: Optional[list] = None,
        mode: str = 'predict',
        **kwargs,
    ):
        """The unified entry for a forward process in both training and test.
        The method should accept only one mode "loss":

        - "loss": Forward and return a dict of losses according to the given
          inputs and data samples.
        Note that this method doesn't handle neither back propagation nor
        optimizer updating, which are done in the :meth:`train_step`.

        Args:
            inputs (dict of torch.Tensor):
                img: pre_processed img tensor  (N, C, ...).
                text: tokenized text (N, L)
            data_samples (List[CaptionDataSample], optional):
            The annotation data of every samples.
                'image': raw image data
                'text' tokenized text
            mode (str): Return what kind of value. Defaults to 'tensor'.

        Returns:
            The return type depends on ``mode``.
            - If ``mode="loss"``, return a dict of tensor.
        """
        if mode == 'loss':
            return self.loss(inputs, data_samples, **kwargs)
        elif mode == 'predict':
            return self.predict(inputs, data_samples, **kwargs)
        else:
            raise RuntimeError(f'Invalid mode "{mode}".')

    def predict(
        self,
        inputs,
        data_samples=None,
        post_process=True,
        **generation_config,
    ):
        images = inputs['imgs']
        input_ids = self.prepare_input_ids(
            images.size(0), data_samples, images.device)

        num_beams = generation_config.get(
            'num_beams', getattr(self.model.generation_config, 'num_beams'))

        decoder_prompts = self.get_decoder_prompts(data_samples)
        constrain_fn = partial(
            apply_constraint,
            constraint_trie=self.constraint_trie,
            decoder_prompts=decoder_prompts,
            num_beams=num_beams,
        )

        outputs = self.model.generate(
            input_ids=input_ids,
            images=images,
            images_mask=inputs.get('images_mask'),
            constrain_fn=constrain_fn,
            **generation_config,
        )

        if decoder_prompts is not None:
            # Remove the prefix decoder prompt.
            for prompt_ids, token in zip(decoder_prompts, outputs):
                token[1:len(prompt_ids) + 1] = self.tokenizer.pad_token_id

        if post_process:
            return self.post_process(outputs, data_samples)
        else:
            return outputs

    def get_decoder_prompts(self, data_samples):
        decoder_prompts = []
        if 'decoder_prompt' not in data_samples[0]:
            return None
        for sample in data_samples:
            prompt = ' ' + sample.get('decoder_prompt')
            prompt_ids = self.tokenizer(prompt, add_special_tokens=False)
            prompt_ids = prompt_ids['input_ids']
            decoder_prompts.append(prompt_ids)
        return decoder_prompts

    def prepare_input_ids(self, batch_size, data_samples, device):
        if self.task == 'caption':
            prompt = self.prompt or ' what does the image describe?'
            prompts = [prompt] * batch_size
            prompts = self.tokenizer(prompts, return_tensors='pt')
            return prompt.input_ids.to(device)
        elif self.task == 'vqa':
            prompts = []
            for sample in data_samples:
                prompt = ' ' + sample.get('question')
                prompts.append(prompt)
            prompts = self.tokenizer(
                prompts, return_tensors='pt', padding=True)
            return prompts.input_ids.to(device)

    def post_process(self, outputs, data_samples):

        decode_tokens = self.tokenizer.batch_decode(
            outputs, skip_special_tokens=True)

        out_data_samples = []
        if data_samples is None:
            data_samples = [None for _ in range(len(decode_tokens))]

        if self.task == 'caption':
            process_fn = CleanCaption(
                lowercase=False, remove_chars=string.punctuation).clean
        else:
            process_fn = lambda x: x.strip()  # noqa: E731

        for data_sample, decode_token in zip(data_samples, decode_tokens):
            if data_sample is None:
                data_sample = DataSample()
            if process_fn is not None:
                decode_token = process_fn(decode_token)
            if self.task == 'caption':
                data_sample.pred_caption = decode_token
            elif self.task == 'vqa':
                data_sample.pred_answer = decode_token
            out_data_samples.append(data_sample)

        return out_data_samples
