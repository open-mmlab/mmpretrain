default_scope = 'mmcls'

default_hooks = dict(
    # optimizer configure
    optimizer=dict(type='OptimizerHook', grad_clip=None),

    # record the time to load data and the time it takes to iterate once
    timer=dict(type='IterTimerHook'),

    # logger configure
    logger=dict(type='LoggerHook', interval=100),

    # Parameter Scheduler
    param_scheduler=dict(type='ParamSchedulerHook'),

    # checkpoint saving
    checkpoint=dict(type='CheckpointHook', interval=1),

    # Sampler for distributed training
    sampler_seed=dict(type='DistSamplerSeedHook'),
)

# Environment configure
env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'),
)

vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='ClsVisualizer', vis_backends=vis_backends, name='visualizer')

# Log level configuration
log_level = 'INFO'

# Load from weight
load_from = None

# resume training
resume = False
