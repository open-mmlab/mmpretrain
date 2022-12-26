_base_ = [
    '../_base_/models/levit-256-p16.py',
    '../_base_/datasets/imagenet_bs256_levit_224.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/imagenet_bs1024_adamw_levit.py'
]

model = dict(
    backbone=dict(
        embed_dim=[128, 256, 384],
        num_heads=[4, 6, 8],
        depth=[2, 3, 4],
        key_dim=[16, 16, 16],
        down_ops=[
            # ('Subsample',key_dim, num_heads, attn_ratio, mlp_ratio, stride)
            ['Subsample', 16, 128 // 16, 4, 2, 2],
            ['Subsample', 16, 256 // 16, 4, 2, 2],
        ]),
    head=dict(in_channels=384, ))
