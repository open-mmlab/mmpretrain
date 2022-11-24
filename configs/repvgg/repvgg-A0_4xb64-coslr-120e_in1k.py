_base_ = [
    '../_base_/models/repvgg-A0_in1k.py',
    '../_base_/datasets/imagenet_bs32_pil_resize.py',
    '../_base_/schedules/imagenet_bs256_coslr.py',
    '../_base_/default_runtime.py'
]

# schedule settings
optim_wrapper = dict(
    paramwise_cfg=dict(
        norm_decay_mult=0.0,
        bias_decay_mult=0.0,
        custom_keys={
            'head': dict(decay_mult=0.0),
        }
    )
)


# schedule settings
param_scheduler = dict(
    type='CosineAnnealingLR', T_max=120, by_epoch=True, begin=0, end=120, convert_to_iter_based=True)

train_cfg = dict(by_epoch=True, max_epochs=120)

# model_wrapper_cfg = dict(type="MMDistributedDataParallel", broadcast_buffers=True,  find_unused_parameters=False)

