_base_ = [
    './tinyvit-5m-224_8xb256_in1k.py',
]

# Model settings
model = dict(
    backbone=dict(arch='tinyvit_11m_224', drop_path_rate=0.1),
    head=dict(in_channels=448))
