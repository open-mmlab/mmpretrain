_base_ = 'cae_vit-base-p16_32xb64-amp-coslr-300e_in1k.py'

# dataset 128 x 16
train_dataloader = dict(batch_size=128, num_workers=16)
