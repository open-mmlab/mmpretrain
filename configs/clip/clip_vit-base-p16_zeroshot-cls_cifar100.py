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
        algorithm_keys=['text'],
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
test_evaluator = dict(type='Accuracy', topk=(1, 5))

# schedule settings
train_cfg = None
val_cfg = None
test_cfg = dict()

# model settings
model = dict(
    type='CLIPZeroShot',
    vision_backbone=dict(
        type='VisionTransformer',
        arch='base',
        img_size=224,
        patch_size=16,
        drop_rate=0.,
        layer_cfgs=dict(act_cfg=dict(type='QuickGELU')),
        pre_norm=True,
    ),
    projection=dict(type='CLIPProjection', in_channels=768, out_channels=512),
    text_backbone=dict(
        type='CLIPTransformer',
        width=512,
        layers=12,
        heads=8,
        attn_mask=True,
    ),
    tokenizer=dict(
        type='AutoTokenizer',
        name_or_path='openai/clip-vit-base-patch16',
        use_fast=False),
    vocab_size=49408,
    transformer_width=512,
    proj_dim=512,
    text_prototype='cifar100',
    text_prompt='openai_cifar100',
    context_length=77,
)
