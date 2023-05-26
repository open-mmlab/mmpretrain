_base_ = './_base_.py'

model = dict(
    backbone=dict(
        stem_channels=64,
        drop_path_rate=0.1,
        stage_blocks=[4, 4, 18, 4],
        groups=[4, 8, 16, 32]))
