_base_ = [
    '../../_base_/datasets/imagenet_bs64_swin_224.py',
    '../../_base_/schedules/imagenet_bs1024_adamw_swin.py',
    '../../_base_/default_runtime.py', 'vit-huge-p14_8xb128-coslr-50e_in1k.py'
]

# optimizer wrapper
# learning rate and layer decay rate are set to 0.004 and 0.75 respectively
optim_wrapper = dict(
    type='DeepSpeedOptimWrapper',
    optimizer=dict(
        type='AdamW', lr=4e-3, weight_decay=0.05, betas=(0.9, 0.999)),
    constructor='LearningRateDecayOptimWrapperConstructor',
    paramwise_cfg=dict(
        layer_decay_rate=0.75,
        custom_keys={
            '.ln': dict(decay_mult=0.0),
            '.bias': dict(decay_mult=0.0),
            '.cls_token': dict(decay_mult=0.0),
            '.pos_embed': dict(decay_mult=0.0)
        }))

# training strategy
# Deepspeed with ZeRO3 + fp16
strategy = dict(
    type='DeepSpeedStrategy',
    fp16=dict(
        enabled=True,
        fp16_master_weights_and_grads=False,
        loss_scale=0,
        loss_scale_window=500,
        hysteresis=2,
        min_loss_scale=1,
        initial_scale_power=15,
    ),
    inputs_to_half=['inputs'],
    zero_optimization=dict(
        stage=3,
        allgather_partitions=True,
        reduce_scatter=True,
        allgather_bucket_size=50000000,
        reduce_bucket_size=50000000,
        overlap_comm=True,
        contiguous_gradients=True,
        cpu_offload=False,
    ))

# runner which supports strategies
runner_type = 'FlexibleRunner'
