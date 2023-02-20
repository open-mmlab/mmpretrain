_base_ = 'vit-large-p16_8xb128-coslr-50e_in1k.py'
# MAE fine-tuning setting

# model settings
# MAE ViT-huge set drop_path_rate to 0.3
model = dict(
    backbone=dict(arch='huge', drop_path_rate=0.3, patch_size=14),
    head=dict(in_channels=1280))
