# Copyright (c) OpenMMLab. All rights reserved.
import string
from typing import Optional

from mmengine.model import BaseModel

from mmpretrain.registry import MODELS, TOKENIZER
from mmpretrain.structures import DataSample
from mmpretrain.datasets import CleanCaption
from .ofa_modules import OFAEncoderDecoder


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
        generation_cfg (dict): The extra generation config, accept the keyword
            arguments of [~`transformers.GenerationConfig`].
            Defaults to an empty dict.
        init_cfg (dict, optional): The initialization config. Defaults to None.
    """
    support_tasks = {'caption'}

    def __init__(
        self,
        encoder_cfg,
        decoder_cfg,
        vocab_size,
        embedding_dim,
        tokenizer,
        task='caption',
        prompt=None,
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

        default_prompts = {
            'caption': ' what does the image describe?',
        }
        self.prompt = prompt or default_prompts.get(task)
        self.task = task
        self.post_process = CleanCaption(lowercase=False).clean

        generation_cfg = {
            'num_beams': 5,
            'max_new_tokens': 16,
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
        prompt = [self.prompt] * images.size(0)
        prompt = self.tokenizer(prompt, return_tensors='pt')

        outputs = self.model.generate(
            input_ids=prompt.input_ids.to(images.device),
            images=images,
            images_mask=inputs.get('images_mask'),
            **generation_config,
        )

        decode_tokens = self.tokenizer.batch_decode(
            outputs, skip_special_tokens=post_process)

        out_data_samples = []
        if data_samples is None:
            data_samples = [None for _ in range(len(decode_tokens))]

        for data_sample, decode_token in zip(data_samples, decode_tokens):
            if data_sample is None:
                data_sample = DataSample()
            if post_process:
                decode_token = self.post_process(decode_token)
            data_sample.pred_caption = decode_token
            out_data_samples.append(data_sample)

        return out_data_samples
