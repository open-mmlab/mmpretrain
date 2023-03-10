_base_ = './maskfeat_mvit-small_16xb32-amp-coslr-300e_k400.py'

optim_wrapper = dict(
    optimizer=dict(
        type='AdamW', lr=8e-4, betas=(0.9, 0.999), weight_decay=0.05))
