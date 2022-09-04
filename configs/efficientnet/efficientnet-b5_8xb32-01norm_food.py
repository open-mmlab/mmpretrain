_base_ = [
    '../_base_/models/efficientnet_b5.py',
    '../_base_/datasets/kt_food_dataset.py',
    '../_base_/schedules/imagenet_bs256_coslr.py',
    '../_base_/default_runtime.py',
]

data=dict(samples_per_gpu=64)
optimizer = dict(type='SGD', lr=0.025, momentum=0.9, weight_decay=0.0001)
model=dict(head=dict(num_classes=50))
fp16 = dict(loss_scale='dynamic')