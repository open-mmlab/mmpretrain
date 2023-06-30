_base_ = ['./vit-huge-p14_8xb128-coslr-50e_in1k.py']

strategy = dict(
    type='FSDPStrategy',
    model_wrapper=dict(
        auto_wrap_policy=dict(
            type='torch.distributed.fsdp.wrap.size_based_auto_wrap_policy',
            min_num_params=1e7)))

optim_wrapper = dict(type='AmpOptimWrapper')

# runner which supports strategies
runner_type = 'FlexibleRunner'
