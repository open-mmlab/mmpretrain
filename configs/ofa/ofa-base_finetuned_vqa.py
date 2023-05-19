_base_ = [
    '../_base_/datasets/coco_vqa.py',
    '../_base_/default_runtime.py',
]

ANS2LABEL = 'https://ofa-beijing.oss-cn-beijing.aliyuncs.com/datasets/vqa_data/trainval_ans2label.pkl'  # noqa: E501

# model settings
model = dict(
    type='OFA',
    task='vqa',
    vocab_size=59457,
    embedding_dim=768,
    ans2label=ANS2LABEL,
    encoder_cfg=dict(
        embed_images=dict(type='OFAResNet', depth=101),
        num_layers=6,
        num_heads=12,
    ),
    decoder_cfg=dict(
        num_layers=6,
        num_heads=12,
    ),
    generation_cfg=dict(
        num_beams=5,
        max_new_tokens=200,
        length_penalty=0.,  # VQA doesn't require longer answer.
        use_cache=True,
    ),
    tokenizer=dict(type='OFATokenizer', name_or_path='OFA-Sys/OFA-base'),
)

# data settings
data_preprocessor = dict(
    type='MultiModalDataPreprocessor',
    mean=[127.5, 127.5, 127.5],
    std=[127.5, 127.5, 127.5],
    to_rgb=True,
)

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='Resize',
        scale=(480, 480),
        interpolation='bicubic',
        backend='pillow'),
    dict(type='OFAAddObjects'),
    dict(
        type='PackInputs',
        algorithm_keys=[
            'question', 'gt_answer', 'gt_answer_weight', 'decoder_prompt'
        ],
        meta_keys=['question_id', 'image_id'],
    ),
]

train_dataloader = None  # Eval only
test_dataloader = dict(dataset=dict(pipeline=test_pipeline))

# schedule settings
train_cfg = None
val_cfg = dict()
test_cfg = dict()
