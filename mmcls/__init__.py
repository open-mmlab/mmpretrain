import mmcv

from .version import __version__, parse_version_info

mmcv_minimum_version = '1.3.8'
mmcv_maximum_version = '1.5.0'
mmcv_version = parse_version_info(mmcv.__version__)


assert (mmcv_version >= parse_version_info(mmcv_minimum_version)
        and mmcv_version <= parse_version_info(mmcv_maximum_version)), \
    f'MMCV=={mmcv.__version__} is used but incompatible. ' \
    f'Please install mmcv>={mmcv_minimum_version}, <={mmcv_maximum_version}.'

__all__ = ['__version__']
