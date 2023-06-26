_base_ = [
    '../_base_/default_runtime.py',
]

zeroshot_prompt = (
    'Question:In which year the value was 51? Short Answer:2014<|endofchunk|>'  # noqa: E501
    'Question:Is the value of Favorable 38 in 2015? Short Answer:Yes<|endofchunk|>'  # noqa: E501
)

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
        norm_cfg=dict(type='LN', eps=1e-5),
        layer_cfgs=dict(act_cfg=dict(type='QuickGELU')),
        final_norm=False,
        out_type='raw',
        pretrained=
        '/mnt/petrelfs/zhaowangbo/openmmlab/vit-large-p14_clip-openai-pre_3rdparty_20230517-95e2af0b.pth',
    ),
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
    task='vqa',
    zeroshot_prompt=zeroshot_prompt,
    final_prompt_tmpl='<image>Question:{question} Short Answer:',
    generation_cfg=dict(num_beams=3, max_new_tokens=5, length_penalty=-2.0))

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
        algorithm_keys=['question', 'gt_answer', 'sub_set'],
        meta_keys=['image_id'],
    ),
]

test_dataloader = dict(
    batch_size=1,
    num_workers=8,
    dataset=dict(
        type='ChartQA',
        data_root='data/chartqa/test',
        data_prefix='png',
        ann_file=['test_human.json', 'test_augmented.json'],
        pipeline=test_pipeline
    ),
    sampler=dict(type='DefaultSampler', shuffle=False),
    persistent_workers=True,
)

test_evaluator = dict(type='ChartQARelaxACC')

# schedule settings
test_cfg = dict()
