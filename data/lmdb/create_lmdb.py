import argparse
import sys

from loguru import logger

from mmcls.datasets.persistences.persist_lmdb import LmdbDataExporter


def parse_args():
    parser = argparse.ArgumentParser(
        description='Making LMDB database with multiprocess')
    parser.add_argument('--i', required=True, help='the input dir of imgs')
    parser.add_argument('--o', required=True, help='output path of LMDB')
    parser.add_argument(
        '--shape',
        default=(256, 256),
        help='reshaping size of imgs before saving')
    parser.add_argument(
        '--batch_size',
        default=100,
        help='batch size of each process to save imgs')

    args = parser.parse_args()

    return args


def main():
    args = parse_args()

    exporter = LmdbDataExporter(
        args.i, args.o, shape=args.shape, batch_size=args.batch_size)

    logger.configure(
        **{'handlers': [
            {
                'sink': sys.stdout,
                'level': 'INFO',
            },
        ]})

    exporter.export()


if __name__ == '__main__':
    main()
