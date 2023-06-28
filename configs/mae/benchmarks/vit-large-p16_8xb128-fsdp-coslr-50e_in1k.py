_base_ = ['./vit-large-p16_8xb128-coslr-50e_in1k.py']

# training strategy
# Deepspeed with ZeRO3 + fp16
strategy = dict(
    type='FSDPStrategy',
    model_wrapper=dict(
        auto_wrap_policy=dict(
            type='torch.distributed.fsdp.wrap.size_based_auto_wrap_policy')))

optim_wrapper = dict(type='AmpOptimWrapper')

# runner which supports strategies
runner_type = 'FlexibleRunner'
