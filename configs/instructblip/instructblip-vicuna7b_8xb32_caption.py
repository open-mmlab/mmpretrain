_base_ = [
    '../_base_/datasets/coco_caption.py',
    '../_base_/default_runtime.py',
]

# model settings
model = dict(
    type='InstructBlipCaption',
    llm_tokenizer=dict(
        type='LlamaTokenizer',
        name_or_path=
        '/mnt/petrelfs/share_data/liuyuan/llm_weights/vicuna_weights_7b'),
    vision_encoder=dict(
        type='BEiTViT',
        # eva-g without the final layer
        arch=dict(
            embed_dims=1408,
            num_layers=39,
            num_heads=16,
            feedforward_channels=6144,
        ),
        img_size=224,
        patch_size=14,
        out_indices=-2,
        layer_scale_init_value=0.0,
        use_abs_pos_emb=True,
        use_rel_pos_bias=False,
        frozen_stages=39,
        final_norm=False,
        use_shared_rel_pos_bias=False,
        out_type='raw',
        pretrained=  # noqa
        'https://download.openmmlab.com/mmpretrain/v1.0/minigpt4/minigpt-4_eva-g-p14_20230615-e908c021.pth'  # noqa
    ),
    text_backbone=dict(
        type='AutoModelForCausalLM',
        name_or_path=
        '/mnt/petrelfs/share_data/liuyuan/llm_weights/vicuna_weights_7b'),
    Qformer=dict(
        type='Qformer',
        model_style='bert-base-uncased',
        vision_model_width=1408,
        add_cross_attention=True,
        cross_attention_freq=2,
        num_query_token=32),
    prompt='Write a short description for the image.',
    max_txt_len=30)

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

# dataset settings
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='Resize',
        scale=(224, 224),
        interpolation='bicubic',
        backend='pillow'),
    dict(type='PackInputs', meta_keys=['image_id']),
]

val_dataloader = dict(dataset=dict(pipeline=test_pipeline))
test_dataloader = val_dataloader
