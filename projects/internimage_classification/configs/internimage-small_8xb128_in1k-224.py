_base_ = './_base_.py'

model = dict(
    backbone=dict(
        stem_channels=80,
        drop_path_rate=0.4,
        stage_blocks=[4, 4, 21, 4],
        groups=[5, 10, 20, 40],
        layer_scale=1e-5,
        post_norm=True),
    head=dict(in_channels=960))
