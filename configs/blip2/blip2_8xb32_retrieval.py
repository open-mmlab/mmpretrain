_base_ = [
    '../_base_/datasets/coco_retrieval.py',
    '../_base_/default_runtime.py',
]

# model settings
model = dict(
    type='Blip2Retrieval',
    tokenizer=dict(type='Blip2Tokenizer', name_or_path='bert-base-uncased'),
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
        layer_scale_init_value=0.0,
        use_abs_pos_emb=True,
        use_rel_pos_bias=False,
        final_norm=False,
        use_shared_rel_pos_bias=False,
        out_type='raw'),
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
        num_classes=256,
    ),
    text_neck=dict(
        type='LinearClsHead',
        in_channels=768,
        num_classes=256,
    ),
    multimodal_head=dict(
        type='ITMHead',
        hidden_size=768,
        with_pooler=False,
    ),
    topk=128,
    max_txt_len=35,
)

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='Resize',
        scale=(364, 364),
        interpolation='bicubic',
        backend='pillow'),
    dict(type='CleanCaption', keys='text'),
    dict(
        type='PackInputs',
        algorithm_keys=['text', 'gt_text_id', 'gt_image_id'],
        meta_keys=['image_id']),
]

val_dataloader = dict(dataset=dict(pipeline=test_pipeline))
test_dataloader = val_dataloader

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
