_base_ = [
    '../_base_/default_runtime.py',
]

# model settings
model = dict(
    type='Otter',
    tokenizer=dict(type='LlamaTokenizer', name_or_path='huggyllama/llama-7b'),
    vision_encoder=dict(
        type='VisionTransformer',
        arch='l',
        patch_size=14,
        pre_norm=True,
        norm_cfg=dict(type='LN', eps=1e-5),
        layer_cfgs=dict(act_cfg=dict(type='mmpretrain.QuickGELU')),
        final_norm=False,
        out_type='raw',
        pretrained=(
            'https://download.openmmlab.com/mmclassification/v0/clip/'
            'vit-large-p14_clip-openai-pre_3rdparty_20230517-95e2af0b.pth'),
    ),
    lang_encoder=dict(
        base=dict(
            type='AutoModelForCausalLM',
            name_or_path='huggyllama/llama-7b',
            local_files_only=True),
        adapter=dict(
            type='FlamingoLMAdapter',
            vis_hidden_size=1024,
            cross_attn_every_n_layers=4,
            use_media_placement_augmentation=False,
            only_attend_previous=True,
        ),
    ),
    task='caption',
    final_prompt_tmpl='<image>User:Please describe the image. GPT:<answer>',
    generation_cfg=dict(
        num_beams=3, max_new_tokens=24, no_repeat_ngram_size=3),
)

# data settings
data_preprocessor = dict(
    type='MultiModalDataPreprocessor',
    mean=[122.770938, 116.7460125, 104.09373615],
    std=[68.5005327, 66.6321579, 70.32316305],
    to_rgb=True,
)

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='ResizeEdge',
        scale=224,
        interpolation='bicubic',
        backend='pillow'),
    dict(type='CenterCrop', crop_size=(224, 224)),
    dict(
        type='PackInputs',
        algorithm_keys=['gt_caption'],
        meta_keys=['image_id'],
    ),
]

val_dataloader = dict(
    batch_size=8,
    num_workers=8,
    dataset=dict(
        type='COCOCaption',
        data_root='data/coco',
        ann_file='annotations/coco_karpathy_val.json',
        pipeline=test_pipeline,
    ),
    sampler=dict(type='DefaultSampler', shuffle=False),
    persistent_workers=True,
)

val_evaluator = dict(
    type='COCOCaption',
    ann_file='data/coco/annotations/coco_karpathy_val_gt.json')

# If you want standard test, please manually configure the test dataset
test_dataloader = val_dataloader
test_evaluator = val_evaluator

# schedule settings
val_cfg = dict()
test_cfg = dict()
