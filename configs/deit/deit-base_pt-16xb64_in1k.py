_base_ = './deit-small_pt-4xb256_in1k.py'

# model settings
model = dict(
    backbone=dict(
        type='VisionTransformer', arch='deit-base', drop_path_rate=0.1),
    head=dict(type='VisionTransformerClsHead', in_channels=768),
)

# data settings
data = dict(samples_per_gpu=64, workers_per_gpu=5)

custom_hooks = [dict(type='EMAHook', momentum=4e-5, priority='ABOVE_NORMAL')]
