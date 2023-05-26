_base_ = './_base_.py'

model = dict(
    backbone=dict(
        stem_channels=112,
        drop_path_rate=0.5,
        stage_blocks=[4, 4, 21, 4],
        groups=[7, 14, 28, 56],
        layer_scale=1e-5,
        post_norm=True),
    head=dict(in_channels=1344))

optim_wrapper = dict(optimizer=dict(lr=0.0005))
