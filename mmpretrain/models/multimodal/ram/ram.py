# Copyright (c) OpenMMLab. All rights reserved.
import os
import pickle
from abc import abstractmethod
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
from mmengine.model import BaseModel

from mmpretrain.registry import MODELS, TOKENIZER
from mmpretrain.structures import DataSample
from .bert import BertConfig, BertLMHeadModel, BertModel
from .openset_utils import build_openset_label_embedding
from .utils import tie_encoder_decoder_weights


def get_path(path):
    file_path = os.path.abspath(os.path.dirname(__file__))
    if not os.path.isabs(path):
        return os.path.join(file_path, path)


class RAM(BaseModel):
    """The implementation of `RAM <https://arxiv.org/abs/2306.03514>`_."""

    def __init__(self,
                 tokenizer: dict,
                 vision_backbone: dict,
                 tag_encoder: dict,
                 tagging_head: dict,
                 text_decoder: dict,
                 device: str = 'cpu',
                 vision_width: int = 1536,
                 prompt='a picture of ',
                 threshold=0.68,
                 delete_tag_index=[],
                 tag_list='./data/ram_tag_list.pickle',
                 tag_list_chinese='./data/ram_tag_list_chinese.pickle',
                 data_preprocessor: Optional[dict] = None,
                 init_cfg: Optional[dict] = None):
        if data_preprocessor is None:
            data_preprocessor = {}
        data_preprocessor.setdefault('type', 'MultiModalDataPreprocessor')
        data_preprocessor = MODELS.build(data_preprocessor)

        super().__init__(
            data_preprocessor=data_preprocessor, init_cfg=init_cfg)

        self.device = device
        # build the visual encoder
        self.visual_encoder = MODELS.build(vision_backbone)

        # build the tokenizer
        self.tokenizer = TOKENIZER.build(tokenizer)
        self.tokenizer.add_special_tokens({'bos_token': '[DEC]'})
        self.tokenizer.add_special_tokens(
            {'additional_special_tokens': ['[ENC]']})
        self.tokenizer.enc_token_id = \
            self.tokenizer.additional_special_tokens_ids[0]

        # build the tag encoder
        # encoder_config = BertConfig.from_json_file(med_config)
        # encoder_config.encoder_width = 512
        encoder_config = BertConfig.from_dict(tag_encoder)
        self.tag_encoder = BertModel(
            config=encoder_config, add_pooling_layer=False)

        # build image-tag-text decoder
        # decoder_config = BertConfig.from_json_file(med_config)
        decoder_config = BertConfig.from_dict(text_decoder)
        self.text_decoder = BertLMHeadModel(config=decoder_config)

        self.delete_tag_index = delete_tag_index
        self.prompt = prompt
        self.prompt_length = len(self.tokenizer(self.prompt).input_ids) - 1

        # load tag list
        self.tag_list = self.load_tag_list(get_path(tag_list))
        self.tag_list_chinese = self.load_tag_list(get_path(tag_list_chinese))

        # create image-tag recognition decoder
        self.threshold = threshold
        self.num_class = len(self.tag_list)
        # q2l_config =  \
        #               BertConfig.from_json_file(f'{CONFIG_PATH}/configs/q2l_config.json')
        # q2l_config.encoder_width = 512
        q2l_config = BertConfig.from_dict(tagging_head)
        self.tagging_head = BertModel(
            config=q2l_config, add_pooling_layer=False)
        self.tagging_head.resize_token_embeddings(len(self.tokenizer))
        self.label_embed = nn.Parameter(
            torch.zeros(self.num_class, q2l_config.encoder_width))

        if q2l_config.hidden_size != 512:
            self.wordvec_proj = nn.Linear(512, q2l_config.hidden_size)
        else:
            self.wordvec_proj = nn.Identity()

        self.fc = nn.Linear(q2l_config.hidden_size, 1)

        self.del_selfattention()

        # share weights of the lowest 2-layer of
        # "image-tag interaction encoder" with
        # the "image-tag recogntion decoder"
        tie_encoder_decoder_weights(self.tag_encoder, self.tagging_head, '',
                                    ' ')
        self.image_proj = nn.Linear(vision_width, 512)
        # self.label_embed = nn.Parameter(torch.load(
        #   f'{CONFIG_PATH}/data/textual_label_embedding.pth',
        #   map_location='cpu').float())

        # adjust thresholds for some tags
        self.class_threshold = torch.ones(self.num_class) * self.threshold
        ram_class_threshold_path = get_path(
            './data/ram_tag_list_threshold.pickle')
        with open(ram_class_threshold_path, 'rb') as f:
            ram_class_threshold = pickle.load(f)
        for key, value in enumerate(ram_class_threshold):
            self.class_threshold[key] = value

    def load_tag_list(self, tag_list_file):
        with open(tag_list_file, 'rb') as f:
            tag_list = pickle.load(f)
        tag_list = np.array(tag_list)
        return tag_list

    # delete self-attention layer of image-tag recognition decoder
    # to reduce computation, follower Query2Label
    def del_selfattention(self):
        del self.tagging_head.embeddings
        for layer in self.tagging_head.encoder.layer:
            del layer.attention

    def get_label_embed(self):
        return torch.nn.functional.relu(self.wordvec_proj(self.label_embed))

    def extract_visual_feature(self, images):
        image_embeds = self.visual_encoder(images)[0]
        image_embeds = image_embeds.flatten(2, 3)
        attn_pool = nn.AdaptiveAvgPool1d(1)
        cls_token = attn_pool(image_embeds).permute(0, 2, 1).contiguous()
        image_embeds = image_embeds.permute(0, 2, 1).contiguous()
        image_embeds = torch.cat([cls_token, image_embeds], dim=1)
        image_embeds = self.image_proj(image_embeds)
        image_atts = torch.ones(
            image_embeds.size()[:-1], dtype=torch.long).to(images.device)
        return image_embeds, image_atts

    def image2tag(self, label_embed, image_embeds, image_atts):
        # recognized image tags using image-tag recogntiion decoder
        # image_cls_embeds = image_embeds[:, 0, :]
        image_spatial_embeds = image_embeds[:, 1:, :]

        bs = image_spatial_embeds.shape[0]
        label_embed = label_embed.unsqueeze(0).repeat(bs, 1, 1)
        tagging_embed = self.tagging_head(
            encoder_embeds=label_embed,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            return_dict=False,
            mode='tagging',
        )

        logits = self.fc(tagging_embed[0]).squeeze(-1)
        return logits

    def forward(
        self,
        images: torch.Tensor,
        data_samples: Optional[list] = None,
        mode: str = 'predict',
        **kwargs,
    ):
        if mode == 'predict':
            return self.predict(images, data_samples, **kwargs)
        else:
            raise RuntimeError(f'Invalid mode "{mode}".')

    @abstractmethod
    def predict(self,
                images: torch.Tensor,
                data_samples: DataSample = None) -> DataSample:
        raise NotImplementedError


@MODELS.register_module()
class RAMNormal(RAM):

    def __init__(self,
                 tokenizer: dict,
                 vision_backbone: dict,
                 tag_encoder: dict,
                 tagging_head: dict,
                 text_decoder: dict,
                 device: str = 'cpu',
                 vision_width: int = 1536,
                 prompt='a picture of ',
                 threshold=0.68,
                 delete_tag_index=[],
                 tag_list='./data/ram_tag_list.pickle',
                 tag_list_chinese='./data/ram_tag_list_chinese.pickle',
                 data_preprocessor: Optional[dict] = None,
                 init_cfg: Optional[dict] = None):
        super().__init__(
            tokenizer,
            vision_backbone,
            tag_encoder,
            tagging_head,
            text_decoder,
            device,
            vision_width,
            prompt,
            threshold,
            delete_tag_index,
            tag_list,
            tag_list_chinese,
            data_preprocessor,
            init_cfg,
        )

    def tag_process(self, logits):
        targets = torch.where(
            torch.sigmoid(logits) > self.class_threshold.to(logits.device),
            torch.tensor(1.0).to(logits.device),
            torch.zeros(self.num_class).to(logits.device))

        tag = targets.cpu().numpy()
        tag[:, self.delete_tag_index] = 0
        tag_output = []
        tag_output_chinese = []
        logits_output = []

        bs = logits.shape[0]
        for b in range(bs):
            index = np.argwhere(tag[b] == 1)
            token = self.tag_list[index].squeeze(axis=1)
            logits_output.append(
                torch.sigmoid(logits)[b][index[:, 0]].cpu().numpy())
            tag_output.append(' | '.join(token))
            token_chinese = self.tag_list_chinese[index].squeeze(axis=1)
            tag_output_chinese.append(' | '.join(token_chinese))

        return [(tag_output, tag_output_chinese), logits_output]

    def predict(self,
                images: torch.Tensor,
                data_samples: DataSample = None) -> DataSample:
        self.eval()
        self.to(self.device)
        images = images.to(self.device)
        label_embed = self.get_label_embed()
        image_embeds, image_atts = self.extract_visual_feature(images)
        logits = self.image2tag(label_embed, image_embeds, image_atts)
        tag_output, logits_output = self.tag_process(logits)
        data_samples.set_field(logits_output, 'logits_output')
        data_samples.set_field(tag_output, 'tag_output')
        return data_samples


@MODELS.register_module()
class RAMOpenset(RAMNormal):

    def __init__(self,
                 tokenizer: dict,
                 vision_backbone: dict,
                 tag_encoder: dict,
                 tagging_head: dict,
                 text_decoder: dict,
                 device: str = 'cpu',
                 vision_width: int = 1536,
                 prompt='a picture of ',
                 threshold=0.68,
                 delete_tag_index=[],
                 tag_list='./data/ram_tag_list.pickle',
                 tag_list_chinese='./data/ram_tag_list_chinese.pickle',
                 data_preprocessor: Optional[dict] = None,
                 init_cfg: Optional[dict] = None):
        super().__init__(
            tokenizer,
            vision_backbone,
            tag_encoder,
            tagging_head,
            text_decoder,
            device,
            vision_width,
            prompt,
            threshold,
            delete_tag_index,
            tag_list,
            tag_list_chinese,
            data_preprocessor,
            init_cfg,
        )

    def set_openset(self,
                    categories: List[str] = None,
                    clip_ckpt: str = '',
                    threshold: float = 0.68):
        openset_label_embedding, openset_categories = \
                            build_openset_label_embedding(
                                categories, clip_ckpt
                            )
        self.tag_list = np.array(openset_categories)
        self.label_embed = nn.Parameter(openset_label_embedding.float())
        self.num_class = len(openset_categories)

        # the threshold for unseen categories is often lower
        self.class_threshold = torch.ones(self.num_class) * threshold

    def tag_process(self, logits):
        targets = torch.where(
            torch.sigmoid(logits) > self.class_threshold.to(logits.device),
            torch.tensor(1.0).to(logits.device),
            torch.zeros(self.num_class).to(logits.device))

        tag = targets.cpu().numpy()
        tag[:, self.delete_tag_index] = 0

        bs = logits.shape[0]
        tag_output = []
        logits_output = []
        for b in range(bs):
            index = np.argwhere(tag[b] == 1)
            token = self.tag_list[index].squeeze(axis=1)
            logits_output.append(
                torch.sigmoid(logits)[b][index[:, 0]].cpu().numpy())
            tag_output.append(' | '.join(token))

        return [(tag_output, [None]), logits_output]
