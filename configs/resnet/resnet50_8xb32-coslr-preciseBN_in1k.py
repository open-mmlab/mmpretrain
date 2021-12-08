_base_ = 'resnet50_8xb32-coslr_in1k.py'

# precise BN hook
custom_hooks = [
    dict(
        type='PreciseBNHook',
        num_items=8192,
        interval=1,
        priority='ABOVE_NORMAL')
]
