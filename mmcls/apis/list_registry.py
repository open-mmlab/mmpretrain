# Copyright (c) OpenMMLab. All rights reserved.


def list_registry(specific_registry=None):
    """List the registry of MMClassification.

    When specific_registry is None, list all available filed of registry

    When specific_registry is given, will list all modules in
        `getattr(registry, specific_registry)`

    Args:
        specific_registry (str|None): which filed of registry need to be listed
    """
    import pprint

    from mmengine.registry import Registry

    from mmcls import registry
    from mmcls.utils import register_all_modules

    register_all_modules()
    valid_registry = dict()
    for attr in dir(registry):
        attr_registry = getattr(registry, attr)
        if isinstance(attr_registry, Registry):
            valid_registry[attr] = attr_registry
        else:
            pass

    if specific_registry is None:
        pprint.pprint(list(valid_registry.keys()))
    else:
        assert specific_registry in valid_registry.keys(), \
            f'{specific_registry} is not a valid registry'
        pprint.pprint(valid_registry[specific_registry]._module_dict)
