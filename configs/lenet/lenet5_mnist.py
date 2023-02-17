# model settings
model = dict(
    type='ImageClassifier',
    backbone=dict(type='LeNet5', num_classes=10),
    neck=None,
    head=dict(
        type='ClsHead',
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
    ))

# dataset settings
dataset_type = 'MNIST'
data_preprocessor = dict(mean=[33.46], std=[78.87], num_classes=10)

pipeline = [dict(type='Resize', scale=32), dict(type='PackClsInputs')]

common_data_cfg = dict(
    type=dataset_type, data_prefix='data/mnist', pipeline=pipeline)

train_dataloader = dict(
    batch_size=128,
    num_workers=2,
    dataset=dict(**common_data_cfg, test_mode=False),
    sampler=dict(type='DefaultSampler', shuffle=True),
)

val_dataloader = dict(
    batch_size=128,
    num_workers=2,
    dataset=dict(**common_data_cfg, test_mode=True),
    sampler=dict(type='DefaultSampler', shuffle=False),
)
val_evaluator = dict(type='Accuracy', topk=(1, ))

test_dataloader = val_dataloader
test_evaluator = val_evaluator

# schedule settings
optim_wrapper = dict(
    optimizer=dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001))

param_scheduler = dict(
    type='MultiStepLR',  # learning policy, decay on several milestones.
    by_epoch=True,  # update based on epoch.
    milestones=[15],  # decay at the 15th epochs.
    gamma=0.1,  # decay to 0.1 times.
)

train_cfg = dict(by_epoch=True, max_epochs=5, val_interval=1)  # train 5 epochs
val_cfg = dict()
test_cfg = dict()

# runtime settings
default_scope = 'mmpretrain'

default_hooks = dict(
    # record the time of every iteration.
    timer=dict(type='IterTimerHook'),
    # print log every 150 iterations.
    logger=dict(type='LoggerHook', interval=150),
    # enable the parameter scheduler.
    param_scheduler=dict(type='ParamSchedulerHook'),
    # save checkpoint per epoch.
    checkpoint=dict(type='CheckpointHook', interval=1),
    # set sampler seed in distributed evrionment.
    sampler_seed=dict(type='DistSamplerSeedHook'),
)

env_cfg = dict(
    # disable cudnn benchmark
    cudnn_benchmark=False,
    # set multi process parameters
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    # set distributed parameters
    dist_cfg=dict(backend='nccl'),
)

log_level = 'INFO'

# load from which checkpoint
load_from = None

# whether to resume the training of the checkpoint
resume_from = None

# NOTE: `auto_scale_lr` is for automatically scaling LR
# based on the actual training batch size.
# base_batch_size = (1 GPUs) x (128 samples per GPU)
auto_scale_lr = dict(base_batch_size=128)
