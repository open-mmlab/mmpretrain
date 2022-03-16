import time
import numpy as np

activate = False
# start time, incremental time, counter
_internal_dict = {}


def reset():
    global _internal_dict
    _internal_dict = {}


def enable():
    global activate
    activate = True


def start(_str):
    if activate:
        if _str not in _internal_dict:
            _internal_dict[_str] = [time.time(), 0, 1]
        else:
            _, deltaT, counter = _internal_dict[_str]
            _internal_dict[_str] = [time.time(), deltaT, counter+1]


def stop(_str):
    if activate:
        if _str not in _internal_dict:
            _internal_dict[_str] = [None, 0, 0]
        else:
            currentT, deltaT, counter = _internal_dict[_str]
            deltaT = time.time() - currentT + deltaT
            _internal_dict[_str] = [None, deltaT, counter]


def show_info():
    for k, v in _internal_dict.items():
        _, deltaT, counter = v
        print(k, deltaT, counter)

