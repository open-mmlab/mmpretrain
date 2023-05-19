_base_ = [
    '../_base_/datasets/coco_vqa.py',
    '../_base_/default_runtime.py',
]

# model settings
model = dict(
    type='Blip2VQA',
    tokenizer=dict(
        type='AutoTokenizer', name_or_path='facebook/opt-2.7b',
        use_fast=False),
    vision_backbone=dict(
        type='BEiTViT',
        # eva-g without the final layer
        arch=dict(
            embed_dims=1408,
            num_layers=39,
            num_heads=16,
            feedforward_channels=6144,
        ),
        img_size=364,
        patch_size=14,
        out_indices=-2,
        layer_scale_init_value=0.0,
        use_abs_pos_emb=True,
        use_rel_pos_bias=False,
        frozen_stages=39,
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

# data settings
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='RandomResizedCrop', scale=224),
    dict(
        type='PackInputs',
        algorithm_keys=['question', 'gt_answer', 'gt_answer_weight'],
        meta_keys=['question_id', 'image_id'],
    ),
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
        algorithm_keys=['question', 'gt_answer', 'gt_answer_weight'],
        meta_keys=['question_id', 'image_id'],
    ),
]

train_dataloader = dict(dataset=dict(pipeline=train_pipeline))
val_dataloader = dict(dataset=dict(pipeline=test_pipeline))
test_dataloader = val_dataloader

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
