# Copyright (c) OpenMMLab. All rights reserved.
import json
import logging
from collections import defaultdict

from mmcv.utils import get_logger


def get_root_logger(log_file=None, log_level=logging.INFO):
    return get_logger('mmcls', log_file, log_level)


def load_json_logs(json_logs):
    """load and convert json_logs to log_dicts.

    Args:
        json_logs (str): paths of json_logs.

    Returns:
        list[dict(int:dict())]: key is epoch, value is a sub dict keys of
            sub dict is different metrics, e.g. memory, bbox_mAP, value of
            sub dict is a list of corresponding values of all iterations.
    """
    log_dicts = [dict() for _ in json_logs]
    for json_log, log_dict in zip(json_logs, log_dicts):
        with open(json_log, 'r') as log_file:
            for line in log_file:
                log = json.loads(line.strip())
                # skip lines without `epoch` field
                if 'epoch' not in log:
                    continue
                epoch = log.pop('epoch')
                if epoch not in log_dict:
                    log_dict[epoch] = defaultdict(list)
                for k, v in log.items():
                    log_dict[epoch][k].append(v)
    return log_dicts
