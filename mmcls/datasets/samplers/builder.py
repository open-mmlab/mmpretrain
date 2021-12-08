from mmcv.utils import Registry, build_from_cfg

SAMPLERS = Registry('sampler')


def build_sampler(cfg, default_args=None):
    if cfg is None:
        return None
    else:
        return build_from_cfg(cfg, SAMPLERS, default_args=default_args)
