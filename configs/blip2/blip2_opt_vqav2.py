_base_ = ['../_base_/default_runtime.py']

# data settings
data_preprocessor = dict(
    type='MultiModalDataPreprocessor',
    mean=[122.770938, 116.7460125, 104.09373615],
    std=[68.5005327, 66.6321579, 70.32316305],
    to_rgb=True,
)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='RandomResizedCrop', scale=224),
    dict(
        type='PackInputs',
        algorithm_keys=['question', 'gt_answer', 'gt_answer_weight']),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='Resize',
        scale=(224, 224),
        interpolation='bicubic',
        backend='pillow'),
    dict(
        type='CleanCaption',
        keys=['question'],
    ),
    dict(
        type='PackInputs',
        algorithm_keys=[
            'question',
            'gt_answer',
            'gt_answer_weight',
        ]),
]

train_dataloader = dict(
    batch_size=16,
    num_workers=8,
    dataset=dict(
        type='COCOVQA',
        data_root='data/coco',
        ann_file='annotations/vqa_val_eval.json',
        data_prefix=dict(img_path='images'),
        pipeline=train_pipeline),
    sampler=dict(type='DefaultSampler', shuffle=True),
    persistent_workers=True,
    drop_last=True,
)

val_dataloader = dict(
    batch_size=64,
    num_workers=8,
    dataset=dict(
        type='COCOVQA',
        data_root='data/coco',
        ann_file='annotations/vqa_val_eval.json',
        data_prefix=dict(img_path='images'),
        pipeline=test_pipeline),
    sampler=dict(type='DefaultSampler', shuffle=False),
    persistent_workers=True,
)
test_dataloader = val_dataloader

val_evaluator = dict(type='VQAAcc')
test_evaluator = val_evaluator

# model settings
model = dict(
    type='BLIP2VQAModel',
    tokenizer=dict(
        type='AutoTokenizer', name_or_path='facebook/opt-2.7b',
        use_fast=False),
    vision_backbone=dict(
        type='BEiTViT',
        arch='eva-g',
        img_size=364,
        patch_size=14,
        out_indices=-2,
        layer_scale_init_value=0.0,
        use_abs_pos_emb=True,
        use_rel_pos_bias=False,
        frozen_stages=40,
        final_norm=False,
        use_shared_rel_pos_bias=False,
        out_type='raw'),
    text_backbone=dict(
        type='OPTForCausalLM', name_or_path='facebook/opt-2.7b'),
    multimodal_backbone=dict(
        type='Qformer',
        model_style='bert-base-uncased',
        vision_model_width=1408,
        add_cross_attention=True,
        cross_attention_freq=2,
        num_query_token=32),
    vision_neck=dict(
        type='LinearClsHead',
        in_channels=768,
        num_classes=2560,
    ),
    prompt='Question: {} Answer:',
    max_txt_len=10)

# schedule settings
optim_wrapper = dict(optimizer=dict(type='AdamW', lr=1e-5, weight_decay=0.05))

param_scheduler = [
    dict(
        type='CosineAnnealingLR',
        by_epoch=True,
        begin=0,
        end=10,
    )
]

train_cfg = dict(max_epochs=10)
val_cfg = dict()
test_cfg = dict()
