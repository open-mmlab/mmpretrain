_base_ = [
    '../_base_/default_runtime.py',
]

data_preprocessor = dict(
    type='MultiModalDataPreprocessor',
    mean=[122.770938, 116.7460125, 104.09373615],
    std=[68.5005327, 66.6321579, 70.32316305],
    to_rgb=True,
)

# dataset settings
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='Resize',
        scale=(224, 224),
        interpolation='bicubic',
        backend='pillow'),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(
        type='CleanCaption',
        keys='chat_content',
        remove_chars='',
        lowercase=False),
    dict(
        type='PackInputs',
        algorithm_keys=['chat_content', 'lang'],
        meta_keys=['image_id']),
]

train_dataloader = dict(
    batch_size=2,
    num_workers=4,
    dataset=dict(
        type='MiniGPT4Dataset',
        data_root='YOUR_DATA_DIRECTORY',
        ann_file='YOUR_DATA_FILE',
        pipeline=train_pipeline),
    sampler=dict(type='DefaultSampler', shuffle=True),
    collate_fn=dict(type='default_collate'),
    drop_last=False,
)

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='Resize',
        scale=(224, 224),
        interpolation='bicubic',
        backend='pillow'),
    dict(type='PackInputs', meta_keys=['image_id']),
]

test_evaluator = dict(
    type='COCOCaption',
    ann_file='data/coco/annotations/coco_karpathy_val_gt.json',
)

test_dataloader = dict(
    batch_size=1,
    dataset=dict(
        type='COCOCaption',
        data_root='data/coco',
        ann_file='annotations/coco_karpathy_val.json',
        pipeline=test_pipeline))

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
        type='AutoModelForCausalLM',
        name_or_path='baichuan-inc/baichuan-7B',
        trust_remote_code=True),
    tokenizer=dict(
        type='AutoTokenizer',
        name_or_path='baichuan-inc/baichuan-7B',
        trust_remote_code=True),
    task='caption',
    prompt_template=dict([('en', '###Ask: {} ###Answer: '),
                          ('zh', '###问：{} ###答：')]),
    raw_prompts=dict([
        ('en', [('<Img><ImageHere></Img> '
                 'Describe this image in detail.'),
                ('<Img><ImageHere></Img> '
                 'Take a look at this image and describe what you notice.'),
                ('<Img><ImageHere></Img> '
                 'Please provide a detailed description of the picture.'),
                ('<Img><ImageHere></Img> '
                 'Could you describe the contents of this image for me?')]),
        ('zh', [('<Img><ImageHere></Img> '
                 '详细描述这张图片。'), ('<Img><ImageHere></Img> '
                                '浏览这张图片并描述你注意到什么。'),
                ('<Img><ImageHere></Img> '
                 '请对这张图片进行详细的描述。'),
                ('<Img><ImageHere></Img> '
                 '你能为我描述这张图片的内容吗？')])
    ]),
    max_txt_len=160,
    end_sym='###')

strategy = dict(
    type='DeepSpeedStrategy',
    fp16=dict(
        enabled=True,
        auto_cast=False,
        fp16_master_weights_and_grads=False,
        loss_scale=0,
        loss_scale_window=1000,
        hysteresis=1,
        min_loss_scale=1,
        initial_scale_power=16,
    ),
    inputs_to_half=[0],
    zero_optimization=dict(
        stage=2,
        allgather_partitions=True,
        allgather_bucket_size=2e8,
        reduce_scatter=True,
        reduce_bucket_size='auto',
        overlap_comm=True,
        contiguous_gradients=True,
    ),
)

# schedule settings
optim_wrapper = dict(
    type='DeepSpeedOptimWrapper',
    optimizer=dict(type='AdamW', lr=1e-3, weight_decay=0.05))

param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=1e-3 / 500,
        by_epoch=False,
        begin=0,
        end=500,
    ),
    dict(
        type='CosineAnnealingLR',
        eta_min=2e-4,
        by_epoch=False,
        begin=500,
    ),
]

train_cfg = dict(by_epoch=True, max_epochs=6)
test_cfg = dict()

runner_type = 'FlexibleRunner'

default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        interval=1,
        by_epoch=True,
        save_last=True,
        max_keep_ckpts=1,
    ))
