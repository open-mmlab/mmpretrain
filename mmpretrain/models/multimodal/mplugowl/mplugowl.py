# Copyright (c) OpenMMLab. All rights reserved.
import random
import re
import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
from mmengine.logging import MMLogger
from mmengine.model import BaseModel

from mmpretrain.registry import MODELS, TOKENIZER
from mmpretrain.structures import DataSample
from transformers.pytorch_utils import find_pruneable_heads_and_indices, prune_linear_layer

def get_ltor_masks_and_position_ids_from_embeddings(data):
    """Build masks and position id for left to right model."""

    # Extract batch size and sequence length.
    micro_batch_size, seq_length = data.size()[:2]

    # Attention mask (lower triangular).
    att_mask_batch = 1
    attention_mask = torch.tril(torch.ones((att_mask_batch, seq_length, seq_length), device=data.device)).view(
        att_mask_batch, 1, seq_length, seq_length
    )

    # Loss mask.
    loss_mask = torch.ones(data.size()[:2], dtype=torch.float, device=data.device)

    # Position ids.
    position_ids = torch.arange(seq_length, dtype=torch.long, device=data.device)
    position_ids = position_ids.unsqueeze(0).expand_as(data[..., 0])

    # Convert attention mask to binary:
    attention_mask = attention_mask < 0.5

    return attention_mask, loss_mask, position_ids


def get_media_indices(my_list):
    if isinstance(my_list, torch.Tensor):
        my_list = my_list.cpu().tolist()
    result = []
    for i in range(len(my_list)):
        if i == 0 and my_list[i] < 0:
            result.append(i)
        elif my_list[i] != my_list[i - 1] and my_list[i] < 0:
            result.append(i)
    return result


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class MplugOwlVisionEmbeddings(BaseModel):
    def __init__(self, hidden_size=1024, image_size=224, patch_size=14, layer_norm_eps=1e-6):
        super().__init__()
        self.hidden_size = hidden_size
        self.image_size = image_size
        self.patch_size = patch_size

        self.cls_token = nn.Parameter(torch.randn(1, 1, self.hidden_size))

        self.patch_embed = nn.Conv2d(
            in_channels=3,
            out_channels=self.hidden_size,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            bias=False,
        )

        self.num_patches = (self.image_size // self.patch_size) ** 2

        self.position_embedding = nn.Parameter(torch.randn(1, self.num_patches + 1, self.hidden_size))

        self.pre_layernorm = LayerNorm(self.hidden_size, eps=layer_norm_eps)

    def forward(self, pixel_values: torch.FloatTensor) -> torch.Tensor:
        batch_size = pixel_values.size(0)
        image_embeds = self.patch_embed(pixel_values)
        image_embeds = image_embeds.flatten(2).transpose(1, 2)

        class_embeds = self.cls_token.expand(batch_size, 1, -1).to(image_embeds.dtype)
        embeddings = torch.cat([class_embeds, image_embeds], dim=1)
        embeddings = embeddings + self.position_embedding[:, : embeddings.size(1)].to(image_embeds.dtype)
        embeddings = self.pre_layernorm(embeddings)
        return embeddings


class MplugOwlVisionAttention(BaseModel):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, hidden_size=1024, num_attention_heads=16, attention_dropout=0.0):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        if self.head_dim * self.num_heads != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size} and `num_heads`:"
                f" {self.num_heads})."
            )
        self.scale = self.head_dim**-0.5
        self.dropout = nn.Dropout(attention_dropout)

        self.query_key_value = nn.Linear(self.hidden_size, 3 * self.hidden_size)
        self.dense = nn.Linear(self.hidden_size, self.hidden_size)

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
        self,
        hidden_states: torch.Tensor,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """Input shape: Batch x Time x Channel"""

        bsz, seq_len, embed_dim = hidden_states.size()

        mixed_qkv = self.query_key_value(hidden_states)

        mixed_qkv = mixed_qkv.reshape(bsz, seq_len, self.num_heads, 3, embed_dim // self.num_heads).permute(
            3, 0, 2, 1, 4
        )  # [3, b, np, sq, hn]
        query_states, key_states, value_states = (
            mixed_qkv[0],
            mixed_qkv[1],
            mixed_qkv[2],
        )

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_states, key_states.transpose(-1, -2))

        attention_scores = attention_scores * self.scale

        # Normalize the attention scores to probabilities.
        attention_probs = torch.softmax(attention_scores, dim=-1)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_states).permute(0, 2, 1, 3)

        new_context_layer_shape = context_layer.size()[:-2] + (self.hidden_size,)
        context_layer = context_layer.reshape(new_context_layer_shape)

        output = self.dense(context_layer)

        outputs = (output, attention_probs) if output_attentions else (output, None)

        return outputs




class MplugOwlMLP(BaseModel):
    def __init__(self, hidden_size=1024, intermediate_size=4096):
        super().__init__()
        self.activation_fn = MODELS.bulid(dict(type="QuickGELU"))
        self.fc1 = nn.Linear(hidden_size, intermediate_size)
        self.fc2 = nn.Linear(intermediate_size, hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = self.fc2(hidden_states)
        return hidden_states


class MplugOwlVisionEncoderLayer(BaseModel):
    def __init__(self, hidden_size=1024, intermediate_size=4096, num_attention_heads=16, attention_dropout=0.0, layer_norm_eps=1e-6):
        super().__init__()
        self.hidden_size = hidden_size
        self.self_attn = MplugOwlVisionAttention(hidden_size, num_attention_heads, attention_dropout)
        self.input_layernorm = LayerNorm(self.hidden_size, eps=layer_norm_eps)
        self.mlp = MplugOwlMLP(hidden_size, intermediate_size)
        self.post_attention_layernorm = LayerNorm(self.hidden_size, eps=layer_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.FloatTensor]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
                `(config.encoder_attention_heads,)`.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
        """
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, attn_weights = self.self_attn(
            hidden_states=hidden_states,
            head_mask=attention_mask,
            output_attentions=output_attentions,
        )
        hidden_states = hidden_states + residual
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)

        hidden_states = hidden_states + residual

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (attn_weights,)

        return outputs


class MplugOwlVisionEncoder(BaseModel):
    """
    Transformer encoder consisting of `config.num_hidden_layers` self attention layers. Each layer is a
    [`MplugOwlVisionEncoderLayer`].

    Args:
        config (`MplugOwlVisionConfig`):
            The corresponding vision configuration for the `MplugOwlEncoder`.
    """

    def __init__(self, hidden_size=1024, intermediate_size=4096, num_attention_heads=16, attention_dropout=0.0,
    layer_norm_eps=1e-6, num_hidden_layers=24):
        super().__init__()
        self.layers = nn.ModuleList([MplugOwlVisionEncoderLayer(hidden_size, intermediate_size, num_attention_heads, attention_dropout, layer_norm_eps) for _ in range(num_hidden_layers)])
        self.gradient_checkpointing = False


    def forward(
        self,
        inputs_embeds,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        r"""
        Args:
            inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
                Embedded representation of the inputs. Should be float, not int tokens.
            attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                [What are attention masks?](../glossary#attention-mask)
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more detail.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        """

        encoder_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        hidden_states = inputs_embeds
        for idx, encoder_layer in enumerate(self.layers):
            if output_hidden_states:
                encoder_states = encoder_states + (hidden_states,)
            if self.gradient_checkpointing and self.training:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs, output_attentions)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(encoder_layer),
                    hidden_states,
                    attention_mask,
                )
            else:
                layer_outputs = encoder_layer(
                    hidden_states,
                    attention_mask,
                    output_attentions=output_attentions,
                )

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        if output_hidden_states:
            encoder_states = encoder_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, encoder_states, all_attentions] if v is not None)
        return (
            hidden_states, encoder_states, all_attentions
        )


@MODELS.register_module()
class MplugOwlVisionModel(BaseModel):

    def __init__(self, hidden_size=1024, image_size=224, patch_size=14, intermediate_size=4096, num_attention_heads=16, attention_dropout=0.0,layer_norm_eps=1e-6, num_hidden_layers=24):
        super().__init__()
        self.hidden_size = hidden_size

        self.embeddings = MplugOwlVisionEmbeddings(hidden_size, image_size, patch_size, layer_norm_eps)
        self.encoder = MplugOwlVisionEncoder(hidden_size, intermediate_size, num_attention_heads, attention_dropout, layer_norm_eps, num_hidden_layers)
        self.post_layernorm = LayerNorm(self.hidden_size, eps=layer_norm_eps)

        self.post_init()


    def forward(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        r"""
        Returns:

        """

        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")

        hidden_states = self.embeddings(pixel_values)

        encoder_outputs = self.encoder(
            inputs_embeds=hidden_states,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        last_hidden_state = encoder_outputs[0]
        last_hidden_state = self.post_layernorm(last_hidden_state)

        pooled_output = last_hidden_state[:, 0, :]
        pooled_output = self.post_layernorm(pooled_output)

        if not return_dict:
            return (last_hidden_state, pooled_output) + encoder_outputs[1:]

        return (
            last_hidden_state=last_hidden_state,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs[1],
            attentions=encoder_outputs[2],
        )

    def get_input_embeddings(self):
        return self.embeddings


class MplugOwlVisualAbstractorMLP(BaseModel):
    def __init__(self, hidden_size=1024, intermediate_size=4096, layer_norm_eps=1e-6):
        super().__init__()
        in_features = hidden_size
        self.act = nn.SiLU()

        self.w1 = nn.Linear(in_features, intermediate_size)
        self.w2 = nn.Linear(intermediate_size, in_features)
        self.w3 = nn.Linear(in_features, intermediate_size)
        self.ffn_ln = LayerNorm(intermediate_size, eps=layer_norm_eps)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.act(self.w1(hidden_states)) * self.w3(hidden_states)
        hidden_states = self.ffn_ln(hidden_states)
        hidden_states = self.w2(hidden_states)
        return hidden_states


class MplugOwlVisualAbstractorMultiHeadAttention(BaseModel):
    def __init__(self, hidden_size=1024, num_attention_heads=16,attention_probs_dropout_prob=0.1, encoder_hidden_size=1024):
        super().__init__()
        if hidden_size % num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention heads (%d)"
                % (hidden_size, num_attention_heads)
            )

        self.num_attention_heads = num_attention_heads
        self.attention_head_size = int(hidden_size / num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(encoder_hidden_size, self.all_head_size)
        self.value = nn.Linear(encoder_hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(attention_probs_dropout_prob)
        self.save_attention = False

    def save_attn_gradients(self, attn_gradients):
        self.attn_gradients = attn_gradients

    def get_attn_gradients(self):
        return self.attn_gradients

    def save_attention_map(self, attention_map):
        self.attention_map = attention_map

    def get_attention_map(self):
        return self.attention_map

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_value=None,
        output_attentions=False,
    ):
        # If this is instantiated as a cross-attention module, the keys
        # and values come from an encoder; the attention mask needs to be
        # such that the encoder's padding tokens are not attended to.
        key_layer = self.transpose_for_scores(self.key(encoder_hidden_states))
        value_layer = self.transpose_for_scores(self.value(encoder_hidden_states))
        attention_mask = encoder_attention_mask

        mixed_query_layer = self.query(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)

        past_key_value = (key_layer, value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        if self.save_attention:
            self.save_attention_map(attention_probs)
            attention_probs.register_hook(self.save_attn_gradients)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs_dropped = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs_dropped = attention_probs_dropped * head_mask

        context_layer = torch.matmul(attention_probs_dropped, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        outputs = outputs + (past_key_value,)
        return outputs


class MplugOwlVisualAbstractorCrossOutput(BaseModel):
    def __init__(self,hidden_size=1024, intermediate_size=4096, layer_norm_eps=1e-6):
        super().__init__()
        dim = hidden_size
        self.out_proj = nn.Linear(dim, dim, bias=True)
        self.norm2 = LayerNorm(dim)
        self.mlp = MplugOwlVisualAbstractorMLP(hidden_size, intermediate_size, layer_norm_eps)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        input_tensor = input_tensor + self.out_proj(hidden_states)
        input_tensor = input_tensor + self.mlp(self.norm2(input_tensor))
        return input_tensor


class MplugOwlVisualAbstractorAttention(BaseModel):
    def __init__(self, hidden_size=1024, num_attention_heads=16, intermediate_size=4096,attention_probs_dropout_prob=0.1,layer_norm_eps=1e-6,encoder_hidden_size=1024):
        super().__init__()
        self.attention = MplugOwlVisualAbstractorMultiHeadAttention(hidden_size, num_attention_heads,attention_probs_dropout_prob, encoder_hidden_size)
        self.output = MplugOwlVisualAbstractorCrossOutput(hidden_size, intermediate_size, layer_norm_eps)
        self.pruned_heads = set()
        self.norm1 = LayerNorm(hidden_size)
        self.normk = LayerNorm(hidden_size)

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        heads, index = find_pruneable_heads_and_indices(
            heads, self.attention.num_attention_heads, self.attention.attention_head_size, self.pruned_heads
        )

        # Prune linear layers
        self.attention.query = prune_linear_layer(self.attention.query, index)
        self.attention.key = prune_linear_layer(self.attention.key, index)
        self.attention.value = prune_linear_layer(self.attention.value, index)
        self.output.dense = prune_linear_layer(self.output.out_proj, index, dim=1)

        # Update hyper params and store pruned heads
        self.attention.num_attention_heads = self.attention.num_attention_heads - len(heads)
        self.attention.all_head_size = self.attention.attention_head_size * self.attention.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor]:
        # HACK we apply norm on q and k
        hidden_states = self.norm1(hidden_states)
        encoder_hidden_states = self.normk(encoder_hidden_states)
        encoder_hidden_states = torch.cat([hidden_states, encoder_hidden_states], dim=1)
        encoder_attention_mask = torch.cat([attention_mask, encoder_attention_mask], dim=-1)
        self_outputs = self.attention(
            hidden_states,
            attention_mask,
            head_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            past_key_value,
            output_attentions,
        )
        attention_output = self.output(self_outputs[0], hidden_states)
        # add attentions if we output them
        outputs = (attention_output,) + self_outputs[1:]
        return outputs


class MplugOwlVisualAbstractorLayer(BaseModel):
    def __init__(self,layer_idx, hidden_size=1024,num_attention_heads=16,intermediate_size=4096,attention_probs_dropout_prob=0.1,layer_norm_eps=1e-6,encoder_hidden_size=1024,chunk_size_feed_forward=None):
        super().__init__()
        self.chunk_size_feed_forward = chunk_size_feed_forward
        self.seq_len_dim = 1

        self.layer_idx = layer_idx

        self.crossattention = MplugOwlVisualAbstractorAttention(hidden_size, num_attention_heads, intermediate_size, attention_probs_dropout_prob, layer_norm_eps, encoder_hidden_size)
        self.has_cross_attention = True

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=False,
    ):
        if encoder_hidden_states is None:
            raise ValueError("encoder_hidden_states must be given for cross-attention layers")
        cross_attention_outputs = self.crossattention(
            hidden_states,
            attention_mask,
            head_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            output_attentions=output_attentions,
        )
        query_attention_output = cross_attention_outputs[0]

        outputs = (query_attention_output,)
        return outputs


class MplugOwlVisualAbstractorEncoder(BaseModel):
    def __init__(self,num_hidden_layers=6, hidden_size=1024,num_attention_heads=16,intermediate_size=4096,attention_probs_dropout_prob=0.1,layer_norm_eps=1e-6,encoder_hidden_size=1024):
        super().__init__()
        self.layers = nn.ModuleList(
            [MplugOwlVisualAbstractorLayer(layer_idx, hidden_size,num_attention_heads,intermediate_size,attention_probs_dropout_prob,layer_norm_eps,encoder_hidden_size) for layer_idx in range(num_hidden_layers)]
        )
        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_values=None,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
    ):
        all_hidden_states = () if output_hidden_states else None

        for i in range(self.config.num_hidden_layers):
            layer_module = self.layers[i]
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_head_mask = head_mask[i] if head_mask is not None else None
            past_key_value = past_key_values[i] if past_key_values is not None else None

            if getattr(self.config, "gradient_checkpointing", False) and self.training:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs, past_key_value, output_attentions)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer_module),
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                )
            else:
                layer_outputs = layer_module(
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    output_attentions,
                )

            hidden_states = layer_outputs[0]

        return BaseModelOutput(
            last_hidden_state=hidden_states,
        )


@MODELS.register_module()
class MplugOwlVisualAbstractorModel(BaseModel):
    def __init__(self, language_hidden_size, num_hidden_layers=6, hidden_size=1024,num_attention_heads=16,intermediate_size=4096,attention_probs_dropout_prob=0.1,layer_norm_eps=1e-6,encoder_hidden_size=1024):
        super().__init__()

        self.language_hidden_size = language_hidden_size
        self.encoder = MplugOwlVisualAbstractorEncoder(num_hidden_layers, hidden_size,num_attention_heads,intermediate_size,attention_probs_dropout_prob,layer_norm_eps,encoder_hidden_size)
        self.visual_fc = torch.nn.Linear(hidden_size, language_hidden_size)
        self.vit_eos = torch.nn.Parameter(torch.randn(1, 1, language_hidden_size))
        nn.init.trunc_normal_(self.vit_eos, mean=0.0, std=self.config.initializer_range)
        self.post_init()

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    def get_extended_attention_mask(
        self,
        attention_mask: torch.Tensor,
        input_shape: Tuple[int],
        device: torch.device,
    ) -> torch.Tensor:
        """
        Makes broadcastable attention and causal masks so that future and masked tokens are ignored.

        Arguments:
            attention_mask (`torch.Tensor`):
                Mask with ones indicating tokens to attend to, zeros for tokens to ignore.
            input_shape (`Tuple[int]`):
                The shape of the input to the model.
            device: (`torch.device`):
                The device of the input to the model.

        Returns:
            `torch.Tensor` The extended attention mask, with a the same dtype as `attention_mask.dtype`.
        """
        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        if attention_mask.dim() == 3:
            extended_attention_mask = attention_mask[:, None, :, :]
        elif attention_mask.dim() == 2:
            # Provided a padding mask of dimensions [batch_size, seq_length]
            # - the model is an encoder, so make the mask broadcastable to [batch_size, num_heads, seq_length, seq_length]
            extended_attention_mask = attention_mask[:, None, None, :]
        else:
            raise ValueError(
                "Wrong shape for input_ids (shape {}) or attention_mask (shape {})".format(
                    input_shape, attention_mask.shape
                )
            )

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.to(dtype=self.dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        return extended_attention_mask

    def forward(
        self,
        query_embeds,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_values=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        encoder_hidden_states  (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, `optional`):
            Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention if
            the model is configured as a decoder.
        encoder_attention_mask (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, `optional`):
            Mask to avoid performing attention on the padding token indices of the encoder input. This mask is used in
            the cross-attention if the model is configured as a decoder. Mask values selected in `[0, 1]`:
            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.
        past_key_values (`tuple(tuple(torch.FloatTensor))` of length `config.n_layers` with each tuple having 4 tensors of:
            shape `(batch_size, num_heads, sequence_length - 1, embed_size_per_head)`): Contains precomputed key and
            value hidden states of the attention blocks. Can be used to speed up decoding. If `past_key_values` are
            used, the user can optionally input only the last `decoder_input_ids` (those that don't have their past key
            value states given to this model) of shape `(batch_size, 1)` instead of all `decoder_input_ids` of shape
            `(batch_size, sequence_length)`.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        embedding_output = query_embeds
        input_shape = embedding_output.size()[:-1]
        batch_size, seq_length = input_shape
        device = embedding_output.device

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        if attention_mask is None:
            attention_mask = torch.ones(
                (query_embeds.shape[0], query_embeds.shape[1]), dtype=torch.long, device=query_embeds.device
            )
        extended_attention_mask = self.get_extended_attention_mask(attention_mask, input_shape, device)

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if encoder_hidden_states is not None:
            if type(encoder_hidden_states) == list:
                encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states[0].size()
            else:
                (
                    encoder_batch_size,
                    encoder_sequence_length,
                    _,
                ) = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)

            if type(encoder_attention_mask) == list:
                encoder_extended_attention_mask = [self.invert_attention_mask(mask) for mask in encoder_attention_mask]
            elif encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
                encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
            else:
                encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            past_key_values=past_key_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = encoder_outputs[0]
        pooled_output = sequence_output[:, 0, :]

        sequence_output = self.visual_fc(sequence_output)
        sequence_output = torch.cat([sequence_output, self.vit_eos.repeat(sequence_output.shape[0], 1, 1)], dim=1)

        return (
            sequence_output,
            pooled_output,
            encoder_outputs.hidden_states,
        )


@MODELS.register_module()
class MplugOwl(BaseModel):
    def __init__(self, vision_encoder, abstractor_model, lang_encoder, num_query_tokens=64):
        super().__init__()

        self.vision_model = MODELS.build(vision_encoder)

        self.abstractor = MODELS.build(abstractor_model)

        self.query_tokens = nn.Parameter(
            torch.zeros(1, num_query_tokens, self.abstractor.language_hidden_size)
        )

        # if config.use_decoder_only_language_model:
        # from llama.modeling_llama import LlamaForCausalLM
        # language_model = AutoModelForCausalLM.from_config(config.text_config)
        # else:
        #     language_model = AutoModelForSeq2SeqLM.from_config(config.text_config)
        self.language_model = MODELS.build(lang_encoder)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.language_model.set_input_embeddings(value)

    def set_output_embeddings(self, new_embeddings):
        self.language_model.set_output_embeddings(new_embeddings)

    def get_output_embeddings(self) -> nn.Module:
        return self.language_model.get_output_embeddings()

    def get_encoder(self):
        return self.language_model.get_encoder()

    def get_decoder(self):
        return self.language_model.get_decoder()

    def _tie_weights(self):
        if not self.config.use_decoder_only_language_model:
            self.language_model.encoder.embed_tokens = self.language_model.shared
            self.language_model.decoder.embed_tokens = self.language_model.shared

    def get_text_features(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        decoder_input_ids: Optional[torch.Tensor] = None,
        decoder_attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if self.config.use_decoder_only_language_model:
            text_outputs = self.language_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        else:
            inputs_embeds = self.language_model.get_input_embeddings()(input_ids)

            text_outputs = self.language_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                decoder_input_ids=decoder_input_ids,
                decoder_attention_mask=decoder_attention_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                labels=labels,
            )

        return text_outputs

    def get_image_features(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        if pixel_values is not None:
            pixel_values = pixel_values.to(self.vision_model.embeddings.cls_token.data.dtype)

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        vision_outputs = self.vision_model(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        return vision_outputs


class MplugOwlForConditionalGeneration(BaseModel):

    def __init__(self, vision_encoder, abstractor_model, lang_encoder,lang_tokenizer,num_query_tokens=64):
        super().__init__()

        self.vision_model = MODELS.build(vision_encoder)

        self.abstractor = MODELS.build(abstractor_model)

        self.query_tokens = nn.Parameter(
            torch.zeros(1, num_query_tokens, self.abstractor.language_hidden_size)
        )

        self.language_model = MODELS.build(lang_encoder)

        self.language_tokenizer = TOKENIZER.build(lang_tokenizer)

        # Initialize weights and apply final processing
        self.post_init()
        self.main_input_name = "input_ids"
        from transformers import GenerationConfig

        self.generation_config = GenerationConfig(
            max_length=512, do_sample=True, top_k=3, pad_token_id=0, unk_token_id=0, bos_token_id=1, eos_token_id=2
        )

    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.language_model.set_input_embeddings(value)

    def set_output_embeddings(self, new_embeddings):
        self.language_model.set_output_embeddings(new_embeddings)

    def get_output_embeddings(self) -> nn.Module:
        return self.language_model.get_output_embeddings()

    def get_encoder(self):
        return self.language_model.get_encoder()

    def get_decoder(self):
        return self.language_model.get_decoder()

    def _tie_weights(self):
        if not self.config.use_decoder_only_language_model:
            self.language_model.encoder.embed_tokens = self.language_model.shared
            self.language_model.decoder.embed_tokens = self.language_model.shared

    def _preprocess_accelerate(self):
        r"""
        Some pre-processing hacks to make the model `accelerate` compatible. Check
        https://github.com/huggingface/transformers/pull/21707 for more details.
        """
        hf_device_map = self.hf_device_map

        if len(hf_device_map) > 1 and "language_model" not in hf_device_map and torch.cuda.device_count() > 1:
            # warn users about unexpected behavior when using multi-GPU + mPLUG-Owl + `accelerate`.
            MMLogger.warning(
                "The `language_model` is not in the `hf_device_map` dictionary and you are running your script"
                " in a multi-GPU environment. this may lead to unexpected behavior when using `accelerate`."
                " Please pass a `device_map` that contains `language_model` to remove this warning."
                " Please refer to https://github.com/huggingface/blog/blob/main/accelerate-large-models.md for"
                " more details on creating a `device_map` for large models.",
            )

        if hasattr(self.language_model, "_hf_hook"):
            self.language_model._hf_hook.io_same_device = True  # For `generate` compatibility


    def forward(
        self,
        pixel_values: torch.FloatTensor,
        input_ids: torch.FloatTensor,
        num_images,
        non_padding_mask: Optional[torch.LongTensor] = None,
        non_media_mask: Optional[torch.LongTensor] = None,
        prompt_mask: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        labels: Optional[torch.LongTensor] = None,
        return_dict: Optional[bool] = None,
    ):
        r"""
        Returns:

        SFT example:

        ```python
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import MplugOwlProcessor, MplugOwlForConditionalGeneration
        >>> import torch

        >>> device = "cuda" if torch.cuda.is_available() else "cpu"

        >>> processor = MplugOwlProcessor.from_pretrained("MAGAer13/mplug-owl-llama-7b")
        >>> model = MplugOwlForConditionalGeneration.from_pretrained(
        ...     "MAGAer13/mplug-owl-llama-7b", torch_dtype=torch.float16
        ... )
        >>> model.to(device)  # doctest: +IGNORE_RESULT

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> prompt = [
        ...     "The following is a conversation between a curious human and AI assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.\nHuman: <image>\nHuman: how many cats are there?\nAI: "
        ... ]
        >>> inputs = processor(images=[image], text=prompt, return_tensors="pt").to(device, torch.float16)

        >>> generated_ids = model.generate(**inputs)
        >>> generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        >>> print(generated_text)
        There are two cats in the image.
        ```"""
        if pixel_values is not None:
            pixel_values = pixel_values.to(self.vision_model.embeddings.cls_token.data.dtype)
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # get text embedding
        text_tokens_ = input_ids.clone()
        batch_size = input_ids.shape[0]
        # labels = text_tokens_[:, 1:].clone().contiguous()

        media_token_indices = [
            # [:-1] since we would not use the last token for embedding
            get_media_indices(text_tokens_[i][:-1])
            for i in range(batch_size)
        ]
        text_tokens_[text_tokens_ < 0] = 1  # Not used
        # text_tokens = text_tokens_[:, :-1].contiguous()
        text_embeds = self.get_input_embeddings()(text_tokens_)  # Temporally Embedding
        if hasattr(self.language_model, 'transformer') and hasattr(self.language_model.transformer, 'word_embeddings_layernorm'):
            text_embeds = self.language_model.transformer.word_embeddings_layernorm(text_embeds)

        if pixel_values is not None:
            image_embeds = self.vision_model(pixel_values, return_dict=True).last_hidden_state

            image_attention_mask = torch.ones(image_embeds.size()[:-1], dtype=torch.long, device=image_embeds.device)
            query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)

            query_features = self.abstractor(
                query_embeds=query_tokens,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_attention_mask,
            )["last_hidden_state"]
            torch.ones(query_features.size()[:-1], dtype=torch.long).to(query_features.device)
            img_seq_length = query_features.shape[1]

        num_images_per_sample = num_images.long().cpu().tolist()

        text_chunk_embeds = []
        img_idx = 0
        for b in range(batch_size):
            start = 0
            result = []
            if len(media_token_indices[b]) > 0:
                for i, pos in enumerate(media_token_indices[b]):
                    if pos > start:
                        result.append(text_embeds[b, start:pos])
                    result.append(query_features[img_idx + i])
                    start = pos + img_seq_length
            if start < text_embeds.shape[1]:
                result.append(text_embeds[b, start:])

            img_idx += num_images_per_sample[b]
            text_chunk_embeds.append(torch.cat(result, dim=0))

        # Actual Input Embeddings
        input_embeds = torch.stack(text_chunk_embeds, dim=0)

        # Create causal mask and position ids
        _, loss_mask, position_ids = get_ltor_masks_and_position_ids_from_embeddings(input_embeds)

        # Calculate the loss_mask
        non_padding_mask = non_padding_mask.long()
        non_media_mask = non_media_mask.long()
        prompt_mask = prompt_mask.long()  # TODO How to deal with prompt mask
        # from icecream import ic
        # non_padding_mask = non_padding_mask[:,:-1]
        # non_media_mask = non_media_mask[:,:-1]
        # prompt_mask = prompt_mask[:,:-1]
        # attention_mask = attention_mask[:,:-1]
        loss_mask = loss_mask[:, :-1]

        loss_mask = loss_mask * non_padding_mask * non_media_mask * prompt_mask
        labels[:, 1:][loss_mask != 1] = -100
        # Forward into GPT
        outputs = self.language_model(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            labels=labels,
            return_dict=return_dict,
            output_attentions=self.config.output_attentions,
        )
        # outputs.loss = (outputs.loss * loss_mask.view(-1)
        #                 ).sum()/loss_mask.sum()
        return outputs

    @torch.no_grad()
    def generate(
        self,
        pixel_values: torch.FloatTensor = None,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        isdecoder=True,
        **generate_kwargs,
    ) -> torch.LongTensor:
        """
        Overrides `generate` function to be able to use the model as a conditional generator.

        Args:
            pixel_values (`torch.FloatTensor` of shape (batch_size, num_channels, height, width)):
                Input images to be processed.
            input_ids (`torch.LongTensor` of shape (batch_size, sequence_length), *optional*):
                The sequence used as a prompt for the generation.
            attention_mask (`torch.LongTensor` of shape (batch_size, sequence_length), *optional*):
                Mask to avoid performing attention on padding token indices

        Returns:
            captions (list): A list of strings of length batch_size * num_captions.
        """
        if pixel_values is not None:
            pixel_values = pixel_values.to(self.vision_model.embeddings.cls_token.data.dtype)

        if input_ids is None:
            return self.language_model.generate(attention_mask=attention_mask, **generate_kwargs)

        if attention_mask is None:
            attention_mask = input_ids.new_ones(*input_ids.shape)

        batch_size = input_ids.size(0)
        media_token_indices = [get_media_indices(input_ids[i]) for i in range(batch_size)]
        num_images_per_sample = [len(x) for x in media_token_indices]
        input_ids = input_ids.clone()  # prevent inplace modify
        input_ids[input_ids < 0] = 0  # Not used

        if hasattr(self, "hf_device_map"):
            # preprocess for `accelerate`
            self._preprocess_accelerate()
        batch_size = input_ids.shape[0]
        # get text embedding
        inputs_embeds = self.get_input_embeddings()(input_ids)
        if hasattr(self.language_model, 'transformer') and hasattr(self.language_model.transformer, 'word_embeddings_layernorm'):
            inputs_embeds = self.language_model.transformer.word_embeddings_layernorm(inputs_embeds)
        # get visual embedding
        if pixel_values is not None:
            pixel_values = pixel_values.to(input_ids.device)
            with torch.no_grad():
                image_embeds = self.vision_model(pixel_values, return_dict=True).last_hidden_state
                image_attention_mask = torch.ones(
                    image_embeds.size()[:-1], dtype=torch.long, device=image_embeds.device
                )
                query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
                query_outputs = self.abstractor(
                    query_embeds=query_tokens,
                    encoder_hidden_states=image_embeds,
                    encoder_attention_mask=image_attention_mask,
                    return_dict=True,
                )
                query_output = query_outputs["last_hidden_state"]
                image_embeds = query_output
            img_seq_length = image_embeds.shape[1]

            # ===================
            # Get actual input embeddings
            # ===================
            text_chunk_embeds = []
            text_chunk_attns = []
            img_idx = 0

            for b in range(batch_size):
                start = 0
                result = []
                result_attn = []
                for i, pos in enumerate(media_token_indices[b]):
                    if pos > start:
                        result.append(inputs_embeds[b, start:pos])
                        result_attn.append(attention_mask[b, start:pos])
                    result.append(image_embeds[img_idx + i])
                    result_attn.append(torch.ones(image_embeds[img_idx + i].shape[0], device=inputs_embeds.device))
                    start = pos + img_seq_length
                if start < inputs_embeds.shape[1]:
                    result.append(inputs_embeds[b, start:])
                    result_attn.append(attention_mask[b, start:])

                img_idx += num_images_per_sample[b]
                text_chunk_embeds.append(torch.cat(result, dim=0))
                text_chunk_attns.append(torch.cat(result_attn, dim=0))
            inputs_embeds = torch.stack(text_chunk_embeds, dim=0)
            attention_mask = torch.stack(text_chunk_attns, dim=0)

        outputs = self.language_model.generate(
            inputs_embeds=inputs_embeds,
            # input_ids=input_ids,
            attention_mask=attention_mask,
            **generate_kwargs,
        )

        return outputs

    def prepare_inputs_for_generation(
        self, input_ids, pixel_values=None, past_key_values=None, attention_mask=None, **model_kwargs
    ):
        input_shape = input_ids.shape
        # if model is used as a decoder in encoder-decoder model, the decoder attention mask is created on the fly
        if attention_mask is None:
            attention_mask = input_ids.new_ones(input_shape)

        # # cut decoder_input_ids if past_key_values is used
        # if past_key_values is not None:
        #     input_ids = input_ids[:, -1:]

        return {
            "input_ids": input_ids,
            "pixel_values": pixel_values,
            "attention_mask": attention_mask,
            "is_decoder": True,
        }


