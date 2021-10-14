# Refer to https://pytorch.org/blog/ml-models-torchvision-v0.9/#classification
# ----------------------------
# -[x] auto_augment='imagenet'
# -[x] batch_size=128 (per gpu)
# -[x] epochs=600
# -[x] opt='rmsprop'
#     -[x] lr=0.064
#     -[x] eps=0.0316
#     -[x] alpha=0.9
#     -[x] weight_decay=1e-05
#     -[x] momentum=0.9
# -[x] lr_gamma=0.973
# -[x] lr_step_size=2
# -[x] nproc_per_node=8
# -[x] random_erase=0.2
# -[x] workers=16 (workers_per_gpu)
# - modify: RandomErasing use RE-M instead of RE-0

_base_ = [
    '../_base_/models/resnet50_logo.py',
    '../_base_/datasets/rp_all_ds.py',
    '../_base_/default_runtime.py'
]

data = dict(
    samples_per_gpu=516,
    workers_per_gpu=8,
)
evaluation = dict(interval=1, metric=['accuracy', 'crossentropy'])
optimizer = dict(type='SGD', lr=0.4, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(policy='step', step=[30, 60, 90])
runner = dict(type='EpochBasedRunner', max_epochs=100)

work_dir = "/home/ubuntu/train_checkpoints/resnet_all"

