# Copyright (c) OpenMMLab. All rights reserved.
import mmengine.dist as dist
import rich.progress as progress

disable_progress_bar = False


def track(sequence, *args, **kwargs):
    if disable_progress_bar:
        return sequence
    else:
        return progress.track(sequence, *args, **kwargs)


def track_on_main_process(sequence, *args, **kwargs):
    if not dist.is_main_process() or disable_progress_bar:
        yield from sequence
    else:
        yield from progress.track(sequence, *args, **kwargs)
