_base_ = [
    '../_base_/datasets/coco_vg_vqa.py',
    '../_base_/default_runtime.py',
]

# model settings
model = dict(
    type='BlipVQAModel',
    tokenizer=dict(type='BLIPTokenizer', name_or_path='bert-base-uncased'),
    vision_backbone=dict(
        type='VisionTransformer',
        arch='b',
        img_size=480,
        patch_size=16,
        out_type='raw'),
    multimodal_backbone=dict(
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
    head=dict(
        type='VQAGenerationHead',
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
        inference_method='rank',  # or 'generate'
        answer_list_path='annotations/vqa_answer_list.json',
    ),
)

# optimizer
optimizer = dict(type='AdamW', lr=2e-5, weight_decay=0.05)
optim_wrapper = dict(type='OptimWrapper', optimizer=optimizer)

# learning rate scheduler
param_scheduler = [dict(type='CosineAnnealingLR', by_epoch=True)]

# [Online Test] dump for official server (eval.ai), no need gt_answer
test_evaluator = dict(
    type='DumpVQAResult', out_file_path='work_dirs/vqa_result.json')

# # [Offline Test] need gt_answer, here we use 'vqa_val.json' as example
# test_evaluator = dict(type='VQAAcc')
# test_dataloader = dict(dataset=dict(ann_file='annotations/vqa_val.json'))

# runtime settings
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=10)
val_cfg = dict()
test_cfg = dict()

randomness = dict(seed=42)
