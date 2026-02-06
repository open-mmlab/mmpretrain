_base_ = './cspnext-s_8xb256-rsb-a1-600e_in1k.py'

work_dir = "/mnt/hdd/maro_work_dir/cspnext-s-dsconv-relu6-600e_in1k"
act_cfg = dict(type='ReLU6', inplace=True)

# import os
# os.environ["OMP_NUM_THREADS"] = "8"
# os.environ["MKL_NUM_THREADS"] = "8"

data_preprocessor = dict(
    type='ClsDataPreprocessor',
    num_classes=1000,
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    to_rgb=True,
)

model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='CSPNeXt',
        arch='P5',
        out_indices=(4, ),
        expand_ratio=0.5,
        deepen_factor=0.33,
        widen_factor=0.5,
        channel_attention=True,
        norm_cfg=dict(type='BN'),
        act_cfg=act_cfg, use_depthwise=True),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='LinearClsHead',
        num_classes=1000,
        in_channels=512,
        loss=dict(
            type='LabelSmoothLoss',
            label_smooth_val=0.1,
            mode='original',
            loss_weight=1.0),
        topk=(1, 5)),
    train_cfg=dict(augments=[
        dict(type='Mixup', alpha=0.2),
        dict(type='CutMix', alpha=1.0)
    ]))

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='RandomResizedCrop', scale=224),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(type='PackInputs'),  # <--- [필수]
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='ResizeEdge', scale=256, edge='short'),
    dict(type='CenterCrop', crop_size=224),
    dict(type='PackInputs'),  # <--- [필수]
]
train_dataloader = dict(
    batch_size=640,
    num_workers=24,
    persistent_workers=True,
    pin_memory=True,
    prefetch_factor=4,
    dataset=dict(
        type='ImageNet',
        data_root='/mnt/hdd/OpenDataLab___ImageNet-1K/raw/ImageNet-1K',
        ann_file='meta/train.txt',
        data_prefix='train/',
        pipeline=train_pipeline), # <--- [추가]
    sampler=dict(type='RepeatAugSampler', shuffle=True)
)

val_dataloader = dict(
    batch_size=32,
    num_workers=32,
    dataset=dict(
        type='ImageNet',
        data_root='/mnt/hdd/OpenDataLab___ImageNet-1K/raw/ImageNet-1K',
        ann_file='meta/val.txt',
        data_prefix='val/',
        pipeline=test_pipeline), # <--- [추가]
sampler=dict(type='DefaultSampler', shuffle=True) # [테스트] RepeatAugSampler 대신 기본으로 먼저 속도 체크
)
val_cfg = dict() 

# 2. 평가 지표 설정 (정확도 Top-1, Top-5 측정)
val_evaluator = dict(type='Accuracy', topk=(1, 5))

# test_dataloader = val_dataloader
# test_evaluator = val_evaluator

env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'),
    # ▼▼▼ [핵심 해결책] ▼▼▼
    # MMEngine이 CPU를 마음대로 제한하지 못하게 설정
    setup_env=dict(
        # opencv 스레드 제한 해제 등
    ) 
)

vis_backends = [
    dict(type='LocalVisBackend'),
    dict(
        type='WandbVisBackend',
        init_kwargs=dict(
            project='csp', # WandB 프로젝트 이름
            name='CSPNeXt-s-dsconv-relu6',      # (선택) 실험 이름
            # entity='my-username'            # (선택) 팀/유저 이름
        )
    )
]

visualizer = dict(
    type='UniversalVisualizer',  # <--- [중요] mmpretrain 전용 비주얼라이저 사용
    vis_backends=vis_backends
)

default_hooks = dict(
    # 기존 설정 유지
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50),
    param_scheduler=dict(type='ParamSchedulerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    
    # [수정 1] Segmentation 전용 훅 제거 -> 일반 VisualizationHook 사용
    visualization=dict(type='VisualizationHook', enable=True),
    
    # [수정 2] CheckpointHook의 저장 기준 변경
    checkpoint=dict(
        type='CheckpointHook',
        interval=1,                # 1 Epoch마다 저장 시도
        save_best='accuracy/top1', # [변경] Segmentation(mIoU) -> Classification(Top-1 Accuracy)
        rule='greater',            # 정확도는 높을수록 좋으므로 greater
        max_keep_ckpts=3,          # Best 모델 최대 3개 유지 (주석에 따라 1로 줄여도 됨)
        save_last=True             # 가장 마지막 Epoch 모델 별도 저장
    )
)

optim_wrapper = dict(
    type='AmpOptimWrapper', # 기본 OptimWrapper 사용
    optimizer=dict(
        type='Lamb',      # 여기에 반드시 type이 있어야 합니다.
        lr=0.005,
        weight_decay=0.02
    ),
    paramwise_cfg=dict(
        bias_decay_mult=0.0,
        norm_decay_mult=0.0
    ),
    accumulative_counts=1
)