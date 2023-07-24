_base_ = '../mae/mae_vit-base-p16_8xb512-amp-coslr-800e_in1k.py'

randomness = dict(seed=2, diff_rank_seed=True)

# dataset config
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='ToPIL', to_rgb=True),
    dict(type='torchvision/Resize', size=224),
    dict(
        type='torchvision/RandomCrop',
        size=224,
        padding=4,
        padding_mode='reflect'),
    dict(type='torchvision/RandomHorizontalFlip', p=0.5),
    dict(type='ToNumpy', to_bgr=True),
    dict(type='PackInputs')
]

train_dataloader = dict(dataset=dict(pipeline=train_pipeline))

# model config
model = dict(
    type='MFF', backbone=dict(type='MFFViT', out_indices=[0, 2, 4, 6, 8, 11]))
