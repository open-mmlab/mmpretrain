# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Optional

import torch
from mmengine.model import BaseModel
from torch import nn
from transformers import BertTokenizer

from mmpretrain.registry import MODELS, TOKENIZER
from mmpretrain.structures import DataSample


@MODELS.register_module()
class InstructBlipCaption(BaseModel):
    """InstructBlip Caption.

    Module for InstructBlip Caption task.

    Args:
        vision_encoder (dict): The config dict for vision backbone.
        text_backbone (dict): The config dict for text backbone.
        Qformer (dict): The config dict for multimodal backbone.
        llm_proj (dict): The config dict for vision neck.
        llm_tokenizer: (Optional[dict]): The config for llm_tokenizer.
            Defaults to None.
        prompt (str): Prompt used for training and eval.
            Defaults to ''.
        max_txt_len (int): Max text length of input text.
        num_captions (int): Number of captions to be generated for each image.
        data_preprocessor (Optional[dict]): The config for preprocessing input
            data. If None or no specified type, it will use
            "MultiModalDataPreprocessor" as type.
            See :class:`MultiModalDataPreprocessor` for more details.
            Defaults to None.
        init_cfg (Optional[dict]): the config to control the initialization.
            Defaults to None.
    """
    _no_split_modules = ['BEiTViT', 'BertLayer']

    def __init__(self,
                 vision_encoder: dict,
                 text_backbone: dict,
                 Qformer: dict,
                 llm_tokenizer: Optional[dict] = None,
                 prompt: str = '',
                 max_txt_len: int = 256,
                 end_sym: str = '\n',
                 num_captions: int = 1,
                 qformer_text_input=True,
                 data_preprocessor: Optional[dict] = None,
                 init_cfg: Optional[dict] = None) -> None:
        if data_preprocessor is None:
            data_preprocessor = {}
        if isinstance(data_preprocessor, dict):
            data_preprocessor.setdefault('type', 'MultiModalDataPreprocessor')
            data_preprocessor = MODELS.build(data_preprocessor)

        super().__init__(
            init_cfg=init_cfg, data_preprocessor=data_preprocessor)

        # build vision model
        vision_encoder_weight = vision_encoder.pop('pretrained', None)
        self.vision_encoder = MODELS.build(vision_encoder)
        self.ln_vision = nn.LayerNorm(self.vision_encoder.embed_dims)

        if vision_encoder_weight is not None:
            from mmengine.runner.checkpoint import load_checkpoint
            load_checkpoint(self.vision_encoder, vision_encoder_weight)

        # build Qformer
        self.tokenizer = BertTokenizer.from_pretrained(
            'bert-base-uncased', truncation_side='left')
        self.tokenizer.add_special_tokens({'bos_token': '[DEC]'})
        self.Qformer = MODELS.build(Qformer)

        if not qformer_text_input:
            self.Qformer.bert.embeddings.word_embeddings = None
            self.Qformer.bert.embeddings.position_embeddings = None
            for layer in self.Qformer.bert.encoder.layer:
                layer.output = None
                layer.intermediate = None
        else:
            self.Qformer.resize_token_embeddings(len(self.tokenizer))
        self.Qformer.cls = None

        # build language model
        self.llm_tokenizer = TOKENIZER.build(llm_tokenizer)
        self.text_backbone = MODELS.build(text_backbone)

        self.llm_tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        self.llm_tokenizer.add_special_tokens({'bos_token': '</s>'})
        self.llm_tokenizer.add_special_tokens({'eos_token': '</s>'})
        self.llm_tokenizer.add_special_tokens({'unk_token': '</s>'})
        # self.llm_tokenizer.pad_token = self.llm_tokenizer.eos_token

        self.text_backbone.resize_token_embeddings(len(self.llm_tokenizer))
        self.eos_token_id = self.llm_tokenizer(
            '\n', add_special_tokens=False).input_ids[0]

        # freeze the text backbone
        for _, param in self.text_backbone.named_parameters():
            param.requires_grad = False

        self.query_tokens = nn.Parameter(
            torch.zeros(1, self.Qformer.bert.config.query_length,
                        self.Qformer.bert.config.hidden_size))
        self.query_tokens.data.normal_(
            mean=0.0, std=self.Qformer.bert.config.initializer_range)

        # build linear projection layer
        self.llm_proj = nn.Linear(self.Qformer.config.hidden_size,
                                  self.text_backbone.config.hidden_size)

        self.prompt = prompt
        self.max_txt_len = max_txt_len
        self.end_sym = end_sym
        self.end_token_id = self.llm_tokenizer.encode(end_sym)[-1]
        self.num_captions = num_captions
        prompt_tokens = self.llm_tokenizer(prompt, return_tensors='pt')
        self.prompt_length = prompt_tokens.attention_mask.sum(1)
        self.qformer_text_input = qformer_text_input

        if hasattr(self, 'register_load_state_dict_post_hook'):
            self.register_load_state_dict_post_hook(self._ignore_llm_keys_hook)

    def forward(
        self,
        images: torch.Tensor,
        data_samples: Optional[List] = None,
        mode: str = 'loss',
    ) -> List[DataSample]:
        """The unified entry for a forward process in both training and test.
        The method should accept two modes: "predict" and "loss":

        - "predict": Forward and return the predictions, which are fully
          processed to a list of :obj:`DataSample`.
        - "loss": Forward and return a dict of losses according to the given
          inputs and data samples.

        Note that this method doesn't handle neither back propagation nor
        optimizer updating, which are done in the :meth:`train_step`.

        Args:
            images (torch.Tensor): pre_processed img tensor  (N, C, ...).
            data_samples (List[DataSample], optional):
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

    def predict(self,
                images: torch.Tensor,
                data_samples: Optional[list] = None,
                **kwargs) -> List[DataSample]:
        """Predict captions from a batch of inputs.

        Args:
            images (torch.Tensor): The input tensor with shape
                (N, C, ...) in general.
            data_samples (List[DataSample], optional): The annotation
                data of every samples. Defaults to None.
            **kwargs: Other keyword arguments accepted by the ``predict``
                method of :attr:`head`.

        Returns:
            List[DataSample]: Return list of data samples.
        """
        self.llm_tokenizer.padding_side = 'left'

        # extract image features from
        image_embeds = self.ln_vision(self.vision_encoder(images)[0])
        image_atts = torch.ones(
            image_embeds.size()[:-1],
            dtype=torch.long,
        ).to(images.device)

        prompt = [self.prompt] * image_embeds.size(0)

        # distill image features to query tokens
        query_tokens = self.query_tokens.expand(image_embeds.size(0), -1, -1)

        if self.qformer_text_input:
            text_Qformer = self.tokenizer(
                prompt,
                padding='longest',
                truncation=True,
                max_length=self.max_txt_len,
                return_tensors='pt',
            ).to(images.device)
            query_atts = torch.ones(
                query_tokens.size()[:-1], dtype=torch.long).to(images.device)
            Qformer_atts = torch.cat([query_atts, text_Qformer.attention_mask],
                                     dim=1)

        if self.qformer_text_input:
            query_outputs = self.Qformer.bert(
                text_Qformer.input_ids,
                attention_mask=Qformer_atts,
                query_embeds=query_tokens,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_atts,
                return_dict=True,
            )

        else:
            query_outputs = self.Qformer.bert(
                query_embeds=query_tokens,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_atts,
                return_dict=True,
            )
        inputs_llama = self.llm_proj(
            query_outputs.last_hidden_state[:, :query_tokens.size(1), :])
        attns_llama = torch.ones(
            inputs_llama.size()[:-1], dtype=torch.long).to(images.device)

        llama_tokens = self.llm_tokenizer(
            prompt, padding='longest', return_tensors='pt').to(images.device)

        inputs_embeds = self.text_backbone.get_input_embeddings()(
            llama_tokens.input_ids)
        inputs_embeds = torch.cat([inputs_llama, inputs_embeds], dim=1)
        attention_mask = torch.cat([attns_llama, llama_tokens.attention_mask],
                                   dim=1)

        outputs = self.text_backbone.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            do_sample=False,
            top_p=0.9,
            temperature=1.,
            num_beams=5,
            max_new_tokens=self.max_txt_len,
            min_length=1,
            repetition_penalty=1.5,
            length_penalty=1.0,
            num_return_sequences=self.num_captions,
        )

        output_text = self.llm_tokenizer.batch_decode(
            outputs[:, self.prompt_length:], skip_special_tokens=True)
        output_text = [text.strip() for text in output_text]

        out_data_samples = []
        if data_samples is None:
            data_samples = [None for _ in range(len(output_text))]

        for data_sample, decode_token in zip(data_samples, output_text):
            if data_sample is None:
                data_sample = DataSample()
            data_sample.pred_caption = decode_token
            out_data_samples.append(data_sample)

        return out_data_samples

    @staticmethod
    def _ignore_llm_keys_hook(module, incompatible_keys):
        """Avoid warning missing keys of the LLM model."""
        import re
        llm_pattern = '^text_backbone'
        for key in list(incompatible_keys.missing_keys):
            if re.match(llm_pattern, key):
                incompatible_keys.missing_keys.remove(key)
