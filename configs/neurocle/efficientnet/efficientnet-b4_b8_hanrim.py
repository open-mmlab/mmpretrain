_base_ = [
    '../_base_/models/efficientnet_b4.py',
    '../_base_/datasets/hanrim_bs8.py',
    '../_base_/schedules/hanrim.py',
    '../_base_/default_runtime.py',
]

model = dict(
    head=dict(
        num_classes={{_base_.data_preprocessor.num_classes}}
    )
)
