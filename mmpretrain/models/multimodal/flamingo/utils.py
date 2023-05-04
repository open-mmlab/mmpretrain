# Copyright (c) OpenMMLab. All rights reserved.
from mmpretrain.registry import MODELS


class ExtendModule:

    def __new__(cls, base: dict, adapter: dict):

        if isinstance(base, dict):
            base = MODELS.build(base)

        adapter_module = MODELS.get(adapter.pop('type'))
        cls.extend_instance(base, adapter_module)
        return adapter_module.extend_init(base, **adapter)

    @classmethod
    def extend_instance(cls, base, mixin):
        """Apply mixins to a class instance after creation."""
        base_cls = base.__class__
        base_cls_name = base.__class__.__name__
        base.__class__ = type(
            base_cls_name, (mixin, base_cls),
            {})  # mixin needs to go first for our forward() logic to work


def getattr_recursive(obj, att):
    """
    Return nested attribute of obj
    Example: getattr_recursive(obj, 'a.b.c') is equivalent to obj.a.b.c
    """
    if att == '':
        return obj
    i = att.find('.')
    if i < 0:
        return getattr(obj, att)
    else:
        return getattr_recursive(getattr(obj, att[:i]), att[i + 1:])


def setattr_recursive(obj, att, val):
    """
    Set nested attribute of obj
    Example: setattr_recursive(obj, 'a.b.c', val)
        is equivalent to obj.a.b.c = val
    """
    if '.' in att:
        obj = getattr_recursive(obj, '.'.join(att.split('.')[:-1]))
    setattr(obj, att.split('.')[-1], val)
