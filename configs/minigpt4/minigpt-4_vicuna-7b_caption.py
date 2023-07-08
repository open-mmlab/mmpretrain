_base_ = [
    '../_base_/datasets/coco_caption.py',
    '../_base_/default_runtime.py',
]

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

val_dataloader = dict(batch_size=1, dataset=dict(pipeline=test_pipeline))
test_dataloader = val_dataloader

# model settings
model = dict(
    type='MiniGPT4',
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
        layer_scale_init_value=0.0,
        frozen_stages=39,
        use_abs_pos_emb=True,
        use_rel_pos_bias=False,
        final_norm=False,
        use_shared_rel_pos_bias=False,
        out_type='raw',
        pretrained=  # noqa
        'https://download.openmmlab.com/mmpretrain/v1.0/minigpt4/minigpt-4_eva-g-p14_20230615-e908c021.pth'  # noqa
    ),
    q_former_model=dict(
        type='Qformer',
        model_style='bert-base-uncased',
        vision_model_width=1408,
        add_cross_attention=True,
        cross_attention_freq=2,
        num_query_token=32,
        pretrained=  # noqa
        'https://download.openmmlab.com/mmpretrain/v1.0/minigpt4/minigpt-4_qformer_20230615-1dfa889c.pth'  # noqa
    ),
    lang_encoder=dict(
        type='AutoModelForCausalLM', name_or_path='YOUR_PATH_TO_VICUNA'),
    tokenizer=dict(type='LlamaTokenizer', name_or_path='YOUR_PATH_TO_VICUNA'),
    task='caption',
    prompt_template='###Human: {} ###Assistant: ',
    raw_prompts=[
        '<Img><ImageHere></Img> Describe this image in detail.',
        '<Img><ImageHere></Img> Take a look at this image and describe what you notice.',  # noqa
        '<Img><ImageHere></Img> Please provide a detailed description of the picture.',  # noqa
        '<Img><ImageHere></Img> Could you describe the contents of this image for me?',  # noqa
    ],
    max_txt_len=160,
    end_sym='###')

# schedule settings
optim_wrapper = dict(optimizer=dict(type='AdamW', lr=1e-5, weight_decay=0.05))

param_scheduler = [
    dict(
        type='CosineAnnealingLR',
        by_epoch=True,
        begin=0,
        end=5,
    )
]

train_cfg = dict(by_epoch=True, max_epochs=5)
val_cfg = dict()
test_cfg = dict()
