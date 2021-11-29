from mmcv.runner.hooks import HOOKS as MMCV_HOOKS
from mmcv.utils import Registry

HOOKS = Registry('hooks', parent=MMCV_HOOKS)


def build_hook(cfg):
    """Build hook."""
    return HOOKS.build(cfg)
