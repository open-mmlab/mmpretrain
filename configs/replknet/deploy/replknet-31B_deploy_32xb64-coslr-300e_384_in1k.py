_base_ = '../replknet-31B_32xb64-coslr-300e_384_in1k.py'

model = dict(backbone=dict(small_kernel_merged=True))
