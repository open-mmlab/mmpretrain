# Inspiration from
# https://github.com/microsoft/Swin-Transformer/blob/main/config.py

# -[x] EPOCHS = 300
# -[x] WARMUP_EPOCHS = 20
# -[x] WEIGHT_DECAY = 0.05
# -[x] BASE_LR = 5e-4
# -[x] WARMUP_LR = 5e-7, ratio=1e-3
# -[x] MIN_LR = 5e-6, ratio=1e-2
# -[x] CLIP_GRAD = 5.0

# -[x] LR_SCHEDULER.NAME = 'cosine'

# -[x] OPTIMIZER.NAME = 'adamw'
# -[x] EPS = 1e-8
# -[x] BETAS = (0.9, 0.999)
# -[x] MOMENTUM = 0.9

_base_ = [
    '../_base_/models/swin_transformer/tiny_224.py',
    '../_base_/datasets/imagenet_bs128_swin_224.py',
    '../_base_/schedules/imagenet_bs1024_adamw_swin.py',
    '../_base_/default_runtime.py'
]
