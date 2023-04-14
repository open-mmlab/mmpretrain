_base_ = 'simmim_swin-base-w6_8xb256-amp-coslr-100e_in1k-192px.py'

# dataset 16 GPUs x 128
train_dataloader = dict(batch_size=128)
