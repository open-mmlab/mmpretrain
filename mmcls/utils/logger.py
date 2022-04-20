# Copyright (c) OpenMMLab. All rights reserved.
import json
import logging
from collections import defaultdict

from mmcv.utils import get_logger


def get_root_logger(log_file=None, log_level=logging.INFO):
    """Get root logger.

    Args:
        log_file (str, optional): File path of log. Defaults to None.
        log_level (int, optional): The level of logger.
            Defaults to :obj:`logging.INFO`.

    Returns:
        :obj:`logging.Logger`: The obtained logger
    """
    return get_logger('mmcls', log_file, log_level)


def load_json_log(json_log):
    """load and convert json_logs to log_dicts.

    Args:
        json_log (str): The path of the json log file.

    Returns:
        dict[int, dict[str, list]]:
            Key is the epoch, value is a sub dict. The keys in each sub dict
            are different metrics, e.g. memory, bbox_mAP, and the value is a
            list of corresponding values in all iterations in this epoch.

            .. code-block:: python

                # An example output
                {
                    1: {'iter': [100, 200, 300], 'loss': [6.94, 6.73, 6.53]},
                    2: {'iter': [100, 200, 300], 'loss': [6.33, 6.20, 6.07]},
                    ...
                }
    """
    log_dict = dict()
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
    return log_dict
