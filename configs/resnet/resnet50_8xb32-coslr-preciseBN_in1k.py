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
