# Copyright (c) OpenMMLab. All rights reserved.
# data settings
test_transforms_cfg = [
    dict(type='Resize', scale=(384, 384), interpolation='bicubic'),
    dict(
        type='mmpretrain.PackInputs',
        algorithm_keys=['text'],
        meta_keys=['image_id', 'scale_factor'],
    ),
]


def get_ram_cfg(mode='normal'):
    assert mode in ['normal', 'openset'], 'mode must "normal" or "openset"'
    model_type = 'RAMNormal' if mode == 'normal' else 'RAMOpenset'
    model_cfg = dict(
        type=model_type,
        tokenizer=dict(
            type='BertTokenizer',
            name_or_path='/public/DATA/qbw/ckpt/bert-base-uncased',
            use_fast=False),
        vision_backbone=dict(
            type='SwinTransformer',
            arch='large',
            img_size=384,
            window_size=12,
        ),
        tag_encoder={
            'architectures': ['BertModel'],
            'attention_probs_dropout_prob': 0.1,
            'hidden_act': 'gelu',
            'hidden_dropout_prob': 0.1,
            'hidden_size': 768,
            'initializer_range': 0.02,
            'intermediate_size': 3072,
            'layer_norm_eps': 1e-12,
            'max_position_embeddings': 512,
            'model_type': 'bert',
            'num_attention_heads': 12,
            'num_hidden_layers': 12,
            'pad_token_id': 0,
            'type_vocab_size': 2,
            'vocab_size': 30524,
            'encoder_width': 512,
            'add_cross_attention': True
        },
        text_decoder={
            'architectures': ['BertModel'],
            'attention_probs_dropout_prob': 0.1,
            'hidden_act': 'gelu',
            'hidden_dropout_prob': 0.1,
            'hidden_size': 768,
            'initializer_range': 0.02,
            'intermediate_size': 3072,
            'layer_norm_eps': 1e-12,
            'max_position_embeddings': 512,
            'model_type': 'bert',
            'num_attention_heads': 12,
            'num_hidden_layers': 12,
            'pad_token_id': 0,
            'type_vocab_size': 2,
            'vocab_size': 30524,
            'encoder_width': 768,
            'add_cross_attention': True
        },
        tagging_head={
            'architectures': ['BertModel'],
            'attention_probs_dropout_prob': 0.1,
            'hidden_act': 'gelu',
            'hidden_dropout_prob': 0.1,
            'hidden_size': 768,
            'initializer_range': 0.02,
            'intermediate_size': 3072,
            'layer_norm_eps': 1e-12,
            'max_position_embeddings': 512,
            'model_type': 'bert',
            'num_attention_heads': 4,
            'num_hidden_layers': 2,
            'pad_token_id': 0,
            'type_vocab_size': 2,
            'vocab_size': 30522,
            'encoder_width': 512,
            'add_cross_attention': True,
            'add_tag_cross_attention': False
        },
        data_preprocessor=dict(
            type='MultiModalDataPreprocessor',
            mean=[0.48145466 * 255, 0.4578275 * 255, 0.40821073 * 255],
            std=[0.26862954 * 255, 0.26130258 * 255, 0.27577711 * 255],
            to_rgb=False,
        ),
    )
    return model_cfg
