# Copyright (c) OpenMMLab. All rights reserved.
import re
from typing import List, Optional

import torch
from mmengine.model import BaseModel

from mmpretrain.registry import MODELS, TOKENIZER
from mmpretrain.structures import DataSample
from .modules import PerceiverResampler
from .utils import ExtendModule


@MODELS.register_module()
class Flamingo(BaseModel):
    """The Open Flamingo model for multiple tasks.

    Args:
        vision_encoder (dict): The config of the vision encoder.
        lang_encoder (dict): The config of the language encoder.
        tokenizer (dict): The tokenizer to encode the text.
        task (int): The task to perform prediction.
        generation_cfg (dict): The extra generation config, accept the keyword
            arguments of [~`transformers.GenerationConfig`].
            Defaults to an empty dict.
        init_cfg (dict, optional): The initialization config. Defaults to None.
    """

    support_tasks = {'caption', 'vqa'}

    def __init__(self,
                 vision_encoder: dict,
                 lang_encoder: dict,
                 tokenizer: dict,
                 task: str = 'caption',
                 generation_cfg: dict = dict(),
                 data_preprocessor: Optional[dict] = None,
                 init_cfg: Optional[dict] = None):
        if data_preprocessor is None:
            data_preprocessor = {}
        data_preprocessor.setdefault('type',
                                     'mmpretrain.MultiModalDataPreprocessor')
        data_preprocessor = MODELS.build(data_preprocessor)

        super().__init__(
            init_cfg=init_cfg, data_preprocessor=data_preprocessor)

        if task not in self.support_tasks:
            raise ValueError(f'Unsupported task {task}, please select '
                             f'the task from {self.support_tasks}.')
        self.task = task

        # init tokenizer
        self.tokenizer = TOKENIZER.build(tokenizer)
        # add Flamingo special tokens to the tokenizer
        self.tokenizer.add_special_tokens(
            {'additional_special_tokens': ['<|endofchunk|>', '<image>']})
        if self.tokenizer.pad_token is None:
            # Issue: GPT models don't have a pad token, which we use to
            # modify labels for the loss.
            self.tokenizer.add_special_tokens({'pad_token': '<PAD>'})

        # init vision encoder related modules
        self.vision_encoder = MODELS.build(vision_encoder)
        self.vision_encoder.init_weights()
        self.perceiver = PerceiverResampler(dim=self.vision_encoder.embed_dims)

        # init language encoder related modules
        self.lang_encoder = ExtendModule(**lang_encoder)
        self.lang_encoder.resize_token_embeddings(len(self.tokenizer))
        self.lang_encoder.media_token_id = self.tokenizer.encode('<image>')[-1]

        # other necessary parameters
        self.eoc_token_id = self.tokenizer.encode('<|endofchunk|>')[-1]
        self.generation_cfg = {
            'num_beams': 1,
            'max_new_tokens': None,
            'temperature': 1.0,
            'top_k': 0,
            'top_p': 1.0,
            'no_repeat_ngram_size': 0,
            'prefix_allowed_tokens_fn': None,
            'length_penalty': 1.0,
            'num_return_sequences': 1,
            'do_sample': False,
            'early_stopping': False,
            **generation_cfg,
        }

    def forward(
        self,
        images: torch.Tensor,
        data_samples: Optional[List[DataSample]] = None,
        mode: str = 'loss',
    ):
        """The unified entry for a forward process in both training and test.
        The method should accept only one mode "loss":

        - "loss": Forward and return a dict of losses according to the given
          inputs and data samples.
        Note that this method doesn't handle neither back propagation nor
        optimizer updating, which are done in the :meth:`train_step`.

        Args:
            inputs (torch.Tensor, tuple): The input tensor with shape
                (N, C, ...) in general.
            data_samples (List[VQADataSample], optional): The annotation
                data of every samples. It's required if ``mode="loss"``.
                Defaults to None.
            mode (str): Return what kind of value. Defaults to 'loss'.

        Returns:
            The return type depends on ``mode``.
            - If ``mode="loss"``, return a dict of tensor.
        """

        if mode == 'loss':
            return self.loss(images, data_samples)
        elif mode == 'predict':
            return self.predict(images, data_samples)
        else:
            raise RuntimeError(f'Invalid mode "{mode}".')

    def extract_vision_feats(self, images: torch.Tensor) -> torch.Tensor:
        """Extract vision features.

        Args:
            images (torch.Tensor): The input images tensor with shape
                (B, T_img, C, H, W) in general. Images in the same chunk
                are collated along T_img. Video data is not supported yet.

        Returns:
            torch.Tensor: Return extracted features.
        """
        b, T = images.shape[:2]
        # b T c h w -> (b T) c h w
        images = images.view(b * T, *images.shape[-3:])

        with torch.no_grad():
            vision_feats = self.vision_encoder(images)[-1][:, 1:]

        # (b T F) v d -> b T F v d  Only support F=1 here
        vision_feats = vision_feats.view(b, T, 1, *vision_feats.shape[-2:])

        vision_feats = self.perceiver(vision_feats)  # reshapes to (b, T, n, d)
        return vision_feats

    def predict(self,
                images: torch.Tensor,
                data_samples: Optional[List[DataSample]] = None,
                **generation_cfg):
        """Predict generation results from a batch of inputs.

        Args:
            images (torch.Tensor): The input images tensor with shape
                (B, T_img, C, H, W) in general. Images in the same chunk
                are collated along T_img. Video data is not supported yet.
            data_samples (List[DataSample], optional): The annotation
                data of every samples. Defaults to None.
            **generation_cfg: Other keyword arguments accepted by the
                ``generate`` method of :attr:`lang_encoder`.

        Returns:
            List[DataSample]: Return list of data samples.
        """
        # generation_cfg in prediction should be dominant
        generation_cfg = {**self.generation_cfg, **generation_cfg}
        num_beams = generation_cfg['num_beams']

        if num_beams > 1:
            images = images.repeat_interleave(num_beams, dim=0)

        # extra vision feats and set as language condition feats
        vision_x = self.extract_vision_feats(images)
        for layer in self.lang_encoder._get_decoder_layers():
            layer.condition_vis_x(vision_x)

        # get nshot prompt for prediction and tokenize
        prompt = [ds.nshot_prompt for ds in data_samples]
        self.tokenizer.padding_side = 'left'
        input_text = self.tokenizer(
            prompt,
            padding='longest',
            truncation=True,
            return_tensors='pt',
            max_length=2000,
        ).to(images.device)

        outputs = self.lang_encoder.generate(
            input_text.input_ids,
            attention_mask=input_text.attention_mask,
            eos_token_id=self.eoc_token_id,
            **generation_cfg)

        # clear conditioned layers for language models
        self.lang_encoder.clear_conditioned_layers()

        # remove prefix
        outputs = outputs[:, len(input_text.input_ids[0]):]

        return self.post_process(outputs, data_samples)

    def post_process(
            self, outputs: torch.Tensor,
            data_samples: Optional[List[DataSample]]) -> List[DataSample]:
        """Perform post process for outputs for different task.

        Args:
            outputs (torch.Tensor): The generated outputs.
            data_samples (List[DataSample], optional): The annotation
                data of every samples

        Returns:
            List[DataSample]: Return list of data samples.
        """
        outputs = self.tokenizer.batch_decode(
            outputs, skip_special_tokens=True)

        if data_samples is None:
            data_samples = [DataSample() for _ in range(len(outputs))]

        for output, data_sample in zip(outputs, data_samples):
            # remove text pattern
            if self.task == 'caption':
                data_sample.pred_caption = re.split('Output', output,
                                                    1)[0].replace('"', '')
            elif self.task == 'vqa':
                data_sample.pred_answer = re.split('Question|Answer', output,
                                                   1)[0]

        return data_samples
