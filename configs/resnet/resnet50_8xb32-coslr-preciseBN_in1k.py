_base_ = 'resnet50_8xb32-coslr_in1k.py'

# precise BN hook
custom_hooks = [
    dict(type='PreciseBNHook', interval=8192, priority='ABOVE_NORMAL')
]

custom_imports = dict(imports=['mmcls.core.hook'], allow_failed_imports=False)