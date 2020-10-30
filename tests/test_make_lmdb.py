import os
import sys

import lmdb
from loguru import logger

from mmcls.datasets.persistences.persist_lmdb import LmdbDataExporter


def test_lmdb_exporter(img_dir='tests/data/make_lmdb',
                       output_path='tests/data/test.lmdb'):
    exporter = LmdbDataExporter(img_dir=img_dir, output_path=output_path)

    assert exporter.img_dir is not None
    assert exporter.output_path is not None

    assert os.path.exists(exporter.img_dir)
    assert os.path.isdir(exporter.img_dir)

    logger.configure(
        **{'handlers': [
            {
                'sink': sys.stdout,
                'level': 'INFO',
            },
        ]})

    exporter.export()

    assert os.path.exists(exporter.output_path)


def test_read_lmdb(ann_file='tests/data/test.lmdb'):
    env = lmdb.open(ann_file)
    txn = env.begin(write=False)
    for key, imgs in txn.cursor():
        key = key.decode()
        assert '###' in key or 'total' in key
        assert imgs is not None
