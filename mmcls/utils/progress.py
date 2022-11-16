# Copyright (c) OpenMMLab. All rights reserved.
import mmengine.dist as dist
import rich.progress as progress


def track_on_main_process(sequence, *args, **kwargs):
    if not dist.is_main_process():
        return sequence

    yield from progress.track(sequence, *args, **kwargs)
