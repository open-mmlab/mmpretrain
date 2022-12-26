_base_ = [
    '../_base_/models/levit-256-p16.py',
    '../_base_/datasets/imagenet_bs256_levit_224.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/imagenet_bs1024_adamw_levit.py'
]

model = dict(
    backbone=dict(
        drop_path=0.1,
        embed_dim=[384, 512, 768],
        num_heads=[6, 9, 12],
        down_ops=[
            # ('Subsample',key_dim, num_heads, attn_ratio, mlp_ratio, stride)
            ['Subsample', 32, 384 // 32, 4, 2, 2],
            ['Subsample', 32, 512 // 32, 4, 2, 2],
        ]),
    head=dict(in_channels=768, ))
