_base_ = ['./resnet50_32xb64-warmup_in1k.py']
model = dict(
    head=dict(
        type='LinearClsHead',
        num_classes=1000,
        in_channels=2048,
        loss=dict(
            type='LabelSmoothLoss',
            loss_weight=1.0,
            label_smooth_val=0.1,
            num_classes=1000),
    ))

# NOTE: `auto_scale_lr` is for automatically scaling LR,
# USER SHOULD NOT CHANGE ITS VALUES.
# base_batch_size = (32 GPUs) x (64 samples per GPU)
auto_scale_lr = dict(base_batch_size=2048)
