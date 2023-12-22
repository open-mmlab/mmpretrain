#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

from typing import List, Optional, Union

import torch
import torch.nn as nn
from transformers import PreTrainedModel

DEFAULT_IMAGE_TOKEN = '<image>'
DEFAULT_IMAGE_PATCH_TOKEN = '<im_patch>'
DEFAULT_IM_START_TOKEN = '<im_start>'
DEFAULT_IM_END_TOKEN = '<im_end>'


class LlavaLlamaForCausalLM(PreTrainedModel):

    def __init__(self,
                 vision_encoder,
                 lang_encoder,
                 mm_hidden_size,
                 use_im_start_end=True,
                 mm_proj_depth=1,
                 im_start_token: Optional[int] = None,
                 im_end_token: Optional[int] = None,
                 im_token_index: int = -200,
                 mm_vision_select_layer: int = -1):
        super().__init__(lang_encoder.config)
        self.vision_tower = vision_encoder
        self.lang_encoder = lang_encoder

        self.use_im_start_end = use_im_start_end
        self.im_start_token = im_start_token
        self.im_end_token = im_end_token
        self.mm_hidden_size = mm_hidden_size
        self.mm_vision_select_layer = mm_vision_select_layer
        self.im_token_index = im_token_index
        self.lang_hidden_size = lang_encoder.config.hidden_size

        if mm_proj_depth == 1:
            # Llava V1
            mm_projector = nn.Linear(self.mm_hidden_size,
                                     self.lang_hidden_size)
            self.lang_encoder.model.add_module('mm_projector', mm_projector)
        elif mm_proj_depth > 1:
            # Llava V1.5
            modules = [nn.Linear(self.mm_hidden_size, self.lang_hidden_size)]
            for _ in range(1, mm_proj_depth):
                modules.append(nn.GELU())
                modules.append(
                    nn.Linear(self.lang_hidden_size, self.lang_hidden_size))
            mm_projector = nn.Sequential(*modules)
            self.lang_encoder.model.add_module('mm_projector', mm_projector)
        elif mm_proj_depth == 0:
            self.lang_encoder.model.add_module('mm_projector', nn.Identity())

        self.post_init()

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        return_dict: Optional[bool] = None,
    ):
        output_attentions = (
            output_attentions if output_attentions is not None else
            self.config.output_attentions)
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else
            self.config.output_hidden_states)
        return_dict = (
            return_dict
            if return_dict is not None else self.config.use_return_dict)

        (input_ids, attention_mask, past_key_values, inputs_embeds,
         labels) = self.forward_vision_tower(input_ids, attention_mask,
                                             past_key_values, labels, images)

        return self.lang_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            labels=labels,
        )

    def prepare_inputs_for_generation(self,
                                      input_ids,
                                      past_key_values=None,
                                      attention_mask=None,
                                      inputs_embeds=None,
                                      **kwargs):
        if past_key_values:
            input_ids = input_ids[:, -1:]

        # if `inputs_embeds` are passed, we only want to use
        # them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {'inputs_embeds': inputs_embeds}
        else:
            model_inputs = {'input_ids': input_ids}

        model_inputs.update({
            'past_key_values': past_key_values,
            'use_cache': kwargs.get('use_cache'),
            'attention_mask': attention_mask,
            'images': kwargs.get('images', None),
        })
        return model_inputs

    def forward_vision_tower(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.LongTensor,
        past_key_values: torch.FloatTensor,
        labels: torch.LongTensor,
        images: Union[torch.FloatTensor, None] = None,
    ):
        if self.vision_tower is None or images is None or input_ids.shape[
                1] == 1:
            if (past_key_values is not None and self.vision_tower is not None
                    and images is not None and input_ids.shape[1] == 1):
                attention_mask = torch.ones(
                    (attention_mask.shape[0],
                     past_key_values[-1][-1].shape[-2] + 1),
                    dtype=attention_mask.dtype,
                    device=attention_mask.device)
            return input_ids, attention_mask, past_key_values, None, labels

        with torch.no_grad():
            # TODO: support variable number of images (single now)
            feats = self.vision_tower(images)
            image_features = feats[-1][:, 1:]

        image_features = self.lang_encoder.model.mm_projector(image_features)

        new_input_embeds = []
        new_labels = [] if labels is not None else None
        new_attn_mask = [] if attention_mask is not None else None
        for batch_idx, cur_input_ids in enumerate(input_ids):
            cur_img = image_features[batch_idx]

            if (cur_input_ids != self.im_token_index).all():
                # multimodal LLM, but the current sample is not multimodal
                new_input_embeds.append(self.embed_tokens(cur_input_ids))
                if labels is not None:
                    new_labels.append(labels[batch_idx])
                if attention_mask is not None:
                    new_attn_mask.append(attention_mask[batch_idx])
                continue

            img_idx = torch.where(cur_input_ids == self.im_token_index)[0][0]
            if self.use_im_start_end:
                cur_new_input_embeds = torch.cat(
                    [
                        self.embed_tokens(cur_input_ids[:img_idx - 1]),
                        self.embed_tokens(cur_input_ids[img_idx - 1:img_idx]),
                        cur_img,
                        self.embed_tokens(
                            cur_input_ids[img_idx + 1:img_idx + 2]),
                        self.embed_tokens(cur_input_ids[img_idx + 2:]),
                    ],
                    dim=0,
                )
            else:
                cur_new_input_embeds = torch.cat(
                    [
                        self.embed_tokens(cur_input_ids[:img_idx]),
                        cur_img,
                        self.embed_tokens(cur_input_ids[img_idx + 1:]),
                    ],
                    dim=0,
                )
            new_input_embeds.append(cur_new_input_embeds)

            if labels is not None:
                cur_new_labels = torch.cat([
                    labels[batch_idx, :img_idx],
                    labels.new_full((cur_img.size(0), ), -100),
                    labels[batch_idx, img_idx + 1:],
                ],
                                           dim=0)
                new_labels.append(cur_new_labels)

            if attention_mask is not None:
                cur_attn_mask = torch.cat([
                    attention_mask[batch_idx, :img_idx],
                    attention_mask.new_full((cur_img.size(0), ), True),
                    attention_mask[batch_idx, img_idx + 1:],
                ],
                                          dim=0)
                new_attn_mask.append(cur_attn_mask)

        inputs_embeds = torch.stack(new_input_embeds, dim=0)
        if labels is not None:
            labels = torch.stack(new_labels, dim=0)
        if attention_mask is not None:
            attention_mask = torch.stack(new_attn_mask, dim=0)

        return None, attention_mask, past_key_values, inputs_embeds, labels

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (tuple(
                past_state.index_select(0, beam_idx)
                for past_state in layer_past), )
        return reordered_past

    def embed_tokens(self, input_ids):
        return self.lang_encoder.model.embed_tokens(input_ids)
