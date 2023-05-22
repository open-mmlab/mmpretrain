_base_ = [
    '../_base_/datasets/nocaps.py',
    '../_base_/default_runtime.py',
]

# model settings
model = dict(
    type='BlipCaption',
    vision_encoder=dict(
        type='VisionTransformer',
        arch='b',
        img_size=384,
        patch_size=16,
        out_type='raw',
    ),
    tokenizer=dict(type='BlipTokenizer', name_or_path='bert-base-uncased'),
    decoder_head=dict(
        type='SeqGenerationHead',
        decoder=dict(
            type='XBertLMHeadDecoder',
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
    ),
    prompt='a picture of ',
    max_txt_len=20,
)

val_cfg = dict()
test_cfg = dict()
