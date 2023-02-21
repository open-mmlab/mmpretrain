_base_ = 'mocov3_vit-base-p16_16xb256-amp-coslr-300e_in1k.py'

train_dataloader = dict(batch_size=64, num_workers=8)

# model settings
model = dict(
    backbone=dict(arch='large'),  # embed_dim = 1024
    neck=dict(in_channels=1024),
)

optim_wrapper = dict(clip_grad=dict(max_norm=5.0, error_if_nonfinite=False))
randomness = dict(seed=0)
