_base_ = './cspnext-s_8xb256-rsb-a1-600e_in1k.py'
work_dir = "./work_dir/cspnext-tiny-relu6-600e_in100"

act_cfg = dict(type='ReLU6', inplace=True)

data_preprocessor = dict(
    type='ClsDataPreprocessor',
    num_classes=100,
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    to_rgb=True,
)

model = dict(
    data_preprocessor=data_preprocessor,
    backbone=dict(deepen_factor=0.167, widen_factor=0.375, 
        act_cfg=act_cfg,
        # use_depthwise=True
        ),
    head=dict(in_channels=384,
    num_classes=100))

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
    batch_size=384,
    num_workers=12,
    dataset=dict(
        type='ImageNet',
        data_root='/mnt/hdd/OpenDataLab___ImageNet-100/raw/MyImagenet',
        ann_file='meta/train.txt',
        data_prefix='train/',
        pipeline=train_pipeline), # <--- [추가]
    sampler=dict(type='RepeatAugSampler', shuffle=True)
)

val_dataloader = dict(
    batch_size=384,
    num_workers=36,
    dataset=dict(
        type='ImageNet',
        data_root='/mnt/hdd/OpenDataLab___ImageNet-100/raw/MyImagenet',
        ann_file='meta/val.txt',
        data_prefix='val/',
        pipeline=test_pipeline), # <--- [추가]
    sampler=dict(type='DefaultSampler', shuffle=False)
)
val_cfg = dict() 

# 2. 평가 지표 설정 (정확도 Top-1, Top-5 측정)
val_evaluator = dict(type='Accuracy', topk=(1, 5))

# test_dataloader = val_dataloader
# test_evaluator = val_evaluator

vis_backends = [
    dict(type='LocalVisBackend'),
    dict(
        type='WandbVisBackend',
        init_kwargs=dict(
            project='csp', # WandB 프로젝트 이름
            name='CSPNeXt-tiny-default',      # (선택) 실험 이름
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
    type='OptimWrapper', # 기본 OptimWrapper 사용
    optimizer=dict(
        type='Lamb',      # 여기에 반드시 type이 있어야 합니다.
        lr=0.005,
        weight_decay=0.02
    ),
    paramwise_cfg=dict(
        bias_decay_mult=0.0,
        norm_decay_mult=0.0
    )
)