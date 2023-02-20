_base_ = 'cae_vit-base-p16_8xb256-amp-coslr-300e_in1k.py'

# dataset 128 x 16
train_dataloader = dict(batch_size=128)
