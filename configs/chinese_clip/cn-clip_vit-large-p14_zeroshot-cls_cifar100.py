_base_ = '../_base_/default_runtime.py'

# data settings
data_preprocessor = dict(
    type='MultiModalDataPreprocessor',
    mean=[0.48145466 * 255, 0.4578275 * 255, 0.40821073 * 255],
    std=[0.26862954 * 255, 0.26130258 * 255, 0.27577711 * 255],
    to_rgb=False,
)

test_pipeline = [
    dict(type='Resize', scale=(224, 224), interpolation='bicubic'),
    dict(
        type='PackInputs',
        meta_keys=['image_id', 'scale_factor'],
    ),
]

train_dataloader = None
test_dataloader = dict(
    batch_size=32,
    num_workers=8,
    dataset=dict(
        type='CIFAR100',
        data_root='data/cifar100',
        split='test',
        pipeline=test_pipeline),
    sampler=dict(type='DefaultSampler', shuffle=False),
)
test_evaluator = dict(type='Accuracy', topk=(1, ))

# schedule settings
train_cfg = None
val_cfg = None
test_cfg = dict()

# model settings
model = dict(
    type='ChineseCLIP',
    vision_backbone=dict(
        type='VisionTransformer',
        arch='large',
        img_size=224,
        patch_size=14,
        norm_cfg=dict(type='LN', eps=1e-5),
        final_norm=True,
        layer_cfgs=dict(act_cfg=dict(type='QuickGELU')),
        pre_norm=True,
        out_type='cls_token',
    ),
    text_backbone=dict(
        type='BertModelCN',
        config=dict(
            vocab_size=21128,
            pad_token_id=0,
            add_type_embeddings=True,
            attention_probs_dropout_prob=0.1,
            hidden_act='gelu',
            hidden_dropout_prob=0.1,
            hidden_size=768,
            initializer_range=0.02,
            intermediate_size=3072,
            max_position_embeddings=512,
            num_attention_heads=12,
            num_hidden_layers=12,
            type_vocab_size=2,
            layer_norm_eps=1e-12)),
    tokenizer=dict(
        type='FullTokenizer',
        vocab_file=  # noqa
        'https://download.openmmlab.com/mmpretrain/v1.0/chinese_clip/vocab.txt'
    ),
    proj_dim=768,
    text_prototype='cifar100',
)
