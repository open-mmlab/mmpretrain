# Directly inherit the entire recipe you want to use.
_base_ = 'mmpretrain::resnet/resnet50_8xb32_in1k.py'

# This line is to import your own modules.
custom_imports = dict(imports='models')

# Modify the backbone to use your own backbone.
_base_['model']['backbone'] = dict(type='ExampleNet', depth=18)
# Modify the in_channels of classifier head to fit your backbone.
_base_['model']['head']['in_channels'] = 512
