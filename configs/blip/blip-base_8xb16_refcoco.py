_base_ = [
    '../_base_/datasets/refcoco.py',
    '../_base_/default_runtime.py',
]

med_config = {
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
    'add_type_embeddings': False,
    'vocab_size': 30524,
    'encoder_width': 768,
    'add_cross_attention': True
}

model = dict(
    type='BlipGrounding',
    visual_encoder=dict(
        type='VisionTransformer',
        arch='b',
        img_size=384,
        patch_size=16,
        out_type='raw',
    ),
    text_encoder=dict(
        type='XBertEncoder',
        med_config=med_config,
    ),
    multimodal_encoder=dict(
        type='XBertEncoder',
        med_config=med_config,
    ),
    tokenizer=dict(type='BlipTokenizer', name_or_path='bert-base-uncased'),
    head=dict(
        type='GroundingHead',
        decoder=dict(
            type='XBertLMHeadDecoder',
            med_config=med_config,
        ),
        box_l1_loss_coeff=4.0,
        box_giou_loss_coeff=2.0,
    ),
)

# schedule settings
optimizer = dict(type='AdamW', lr=1.5e-5, weight_decay=0.02)
optim_wrapper = dict(type='OptimWrapper', optimizer=optimizer)
param_scheduler = [dict(type='CosineAnnealingLR', by_epoch=True)]

train_cfg = dict(by_epoch=True, max_epochs=120)
val_cfg = dict()
test_cfg = dict()
