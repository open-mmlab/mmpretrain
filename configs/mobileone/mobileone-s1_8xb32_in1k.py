_base_ = [
    '../_base_/models/mobileone/mobileone_s1.py',
    '../_base_/datasets/imagenet_bs32_pil_resize.py',
    '../_base_/schedules/imagenet_bs256_coslr_coswd_300e.py',
    '../_base_/default_runtime.py'
]

# schedule settings
optim_wrapper = dict(paramwise_cfg=dict(norm_decay_mult=0.))

val_dataloader = dict(batch_size=256)
test_dataloader = dict(batch_size=256)

bgr_mean = _base_.data_preprocessor['mean'][::-1]
base_train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='RandomResizedCrop', scale=224, backend='pillow'),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(
        type='RandAugment',
        policies='timm_increasing',
        num_policies=2,
        total_level=10,
        magnitude_level=7,
        magnitude_std=0.5,
        hparams=dict(pad_val=[round(x) for x in bgr_mean])),
    dict(type='PackInputs')
]

import copy  # noqa: E402

# modify start epoch's RandomResizedCrop.scale to 160
train_pipeline_1e = copy.deepcopy(base_train_pipeline)
train_pipeline_1e[1]['scale'] = 160
train_pipeline_1e[3]['magnitude_level'] *= 0.1
_base_.train_dataloader.dataset.pipeline = train_pipeline_1e

# modify 37 epoch's RandomResizedCrop.scale to 192
train_pipeline_37e = copy.deepcopy(base_train_pipeline)
train_pipeline_37e[1]['scale'] = 192
train_pipeline_1e[3]['magnitude_level'] *= 0.2

# modify 112 epoch's RandomResizedCrop.scale to 224
train_pipeline_112e = copy.deepcopy(base_train_pipeline)
train_pipeline_112e[1]['scale'] = 224
train_pipeline_1e[3]['magnitude_level'] *= 0.3

custom_hooks = [
    dict(
        type='SwitchRecipeHook',
        schedule=[
            dict(action_epoch=37, pipeline=train_pipeline_37e),
            dict(action_epoch=112, pipeline=train_pipeline_112e),
        ]),
    dict(
        type='EMAHook',
        momentum=5e-4,
        priority='ABOVE_NORMAL',
        update_buffers=True)
]
