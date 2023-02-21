_base_ = 'simmim_swin-base-w6_16xb128-amp-coslr-800e_in1k-192px.py'

# model settings
model = dict(
    backbone=dict(
        arch='large',
        stage_cfgs=dict(block_cfgs=dict(window_size=12)),
        pad_small_map=True),
    neck=dict(type='SimMIMNeck', in_channels=192 * 2**3, encoder_stride=32),
    head=dict(
        type='SimMIMHead',
        patch_size=4,
        loss=dict(type='SimMIMReconstructionLoss', encoder_in_channels=3)))
