_base_ = [
    '../_base_/datasets/flickr30k_retrieval.py',
    '../_base_/default_runtime.py',
]

# model settings
model = dict(
    type='BlipRetrieval',
    tokenizer=dict(type='BlipTokenizer', name_or_path='bert-base-uncased'),
    vision_backbone=dict(
        type='VisionTransformer',
        arch='b',
        img_size=384,
        patch_size=16,
        out_type='raw',
    ),
    text_backbone=dict(
        type='XBertEncoder',
        med_config=dict(
            architectures=['BertModel'],
            attention_probs_dropout_prob=0.1,
            hidden_act='gelu',
            hidden_dropout_prob=0.1,
            hidden_size=768,
            initializer_range=0.02,
            intermediate_size=3072,
            layer_norm_eps=1e-12,
            max_position_embeddings=512,
            model_type='bert',
            num_attention_heads=12,
            num_hidden_layers=12,
            pad_token_id=0,
            add_type_embeddings=False,
            vocab_size=30524,
            encoder_width=768,
            add_cross_attention=True),
    ),
    vision_neck=dict(
        type='Linear',
        in_features=768,
        out_features=256,
    ),
    text_neck=dict(
        type='Linear',
        in_features=768,
        out_features=256,
    ),
    head=dict(
        type='ITCHead',
        embed_dim=256,
    ),
    multimodal_head=dict(
        type='ITMHead',
        hidden_size=768,
        with_pooler=False,
    ),
    topk=256,
    max_txt_len=35,
)

# optimizer
optimizer = dict(type='AdamW', lr=2e-5, weight_decay=0.04)
optim_wrapper = dict(type='OptimWrapper', optimizer=optimizer)

# learning rate scheduler
param_scheduler = [dict(type='CosineAnnealingLR', by_epoch=True)]

# runtime settings
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=6)
val_cfg = dict(type='RetrievalValLoop')
test_cfg = dict(type='RetrievalTestLoop')

randomness = dict(seed=42)

default_hooks = dict(logger=dict(interval=1))

custom_hooks = [
    dict(
        type='WarmupParamHook',
        param_name='alpha',
        module_name='head',
        warmup_epochs=2)
]
