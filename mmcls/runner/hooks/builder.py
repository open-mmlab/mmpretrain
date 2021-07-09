from mmcv.runner import HOOKS as _HOOKS
from mmcv.utils import Registry

HOOKS = Registry('hook', parent=_HOOKS)
