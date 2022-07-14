_base_ = 'resnet50_8xb32-coslr_in1k.py'

# Precise BN hook will update the bn stats, so this hook should be executed
# before CheckpointHook, which has priority of 'NORMAL'. So set the
# priority of PreciseBNHook to 'ABOVE_NORMAL' here.
custom_hooks = [
    dict(
        type='PreciseBNHook',
        num_samples=8192,
        interval=1,
        priority='ABOVE_NORMAL')
]

# NOTE: `auto_scale_lr` is for automatically scaling LR,
# USER SHOULD NOT CHANGE ITS VALUES.
# base_batch_size = (8 GPUs) x (32 samples per GPU)
auto_scale_lr = dict(base_batch_size=256)
