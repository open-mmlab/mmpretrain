_base_ = [
    '../_base_/default_runtime.py',
]

# model settings
model = dict(
    type='Flamingo',
    tokenizer=dict(
        type='LlamaTokenizer', name_or_path='decapoda-research/llama-7b-hf'),
    vision_encoder=dict(
        type='VisionTransformer',
        arch='l',
        patch_size=14,
        pre_norm=True,
        init_cfg=dict(type='Pretrained', checkpoint='clip_vit_l_14.pth'),
        norm_cfg=dict(type='LN', eps=1e-5),
        layer_cfgs=dict(act_cfg=dict(type='QuickGELU')),
        final_norm=False,
        out_type='raw'),
    lang_encoder=dict(
        base=dict(
            type='AutoModelForCausalLM',
            name_or_path='decapoda-research/llama-7b-hf',
            local_files_only=True),
        adapter=dict(
            type='FlamingoLMAdapter',
            vis_hidden_size=1024,
            cross_attn_every_n_layers=4,
            use_media_placement_augmentation=False),
    ),
    task='caption',
    generation_cfg=dict(num_beams=3, max_new_tokens=20, length_penalty=-2.0))

# data settings
data_preprocessor = dict(
    type='MultiModalDataPreprocessor',
    mean=[122.770938, 116.7460125, 104.09373615],
    std=[68.5005327, 66.6321579, 70.32316305],
    to_rgb=True,
)

test_pipeline = [
    dict(
        type='ApplyToList',
        # Flamingo requires to load multiple images during few-shot inference.
        scatter_key='img_path',
        transforms=[
            dict(type='LoadImageFromFile'),
            dict(
                type='ResizeEdge',
                scale=224,
                interpolation='bicubic',
                backend='pillow'),
            dict(type='CenterCrop', crop_size=(224, 224)),
        ],
        collate_keys=['img', 'scale_factor', 'ori_shape'],
    ),
    dict(
        type='PackInputs',
        algorithm_keys=('text', 'image_id', 'nshot_prompt')),
]

val_dataloader = dict(
    batch_size=8,
    num_workers=8,
    dataset=dict(
        type='FlamingoEvalCOCOCaption',
        data_root='data/coco',
        ann_file='annotations/captions_train2014.json',
        pipeline=test_pipeline,
        num_shots=0,
        num_support_examples=2048,
        num_query_examples=5000,
    ),
    sampler=dict(type='DefaultSampler', shuffle=False),
    persistent_workers=True,
)

val_evaluator = dict(
    type='COCOCaption',
    ann_file='data/coco/annotations/captions_train2014.json')

# If you want standard test, please manually configure the test dataset
test_dataloader = val_dataloader
test_evaluator = val_evaluator

# schedule settings
val_cfg = dict()
test_cfg = dict()
