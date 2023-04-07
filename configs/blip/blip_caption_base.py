_base_ = [
    '../_base_/datasets/coco_caption.py',
    '../_base_/models/blip/caption_base.py',
    '../_base_/default_runtime.py',
]

# optimizer
optimizer = dict(type='AdamW', lr=1e-5, weight_decay=0.05)
optim_wrapper = dict(type='OptimWrapper', optimizer=optimizer)

param_scheduler = [
    dict(
        type='CosineAnnealingLR',
        by_epoch=True,
        begin=0,
        end=10,
    )
]

# runtime settings
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=10)
val_cfg = dict()
test_cfg = dict()
