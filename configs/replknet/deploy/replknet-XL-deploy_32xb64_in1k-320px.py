_base_ = '../replknet-XL_32xb64_in1k-320px.py'

model = dict(backbone=dict(small_kernel_merged=True))
