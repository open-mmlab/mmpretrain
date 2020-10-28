import click
from mmcls.datasets.persistences.persist_lmdb import IdtLmdbDataExporter


# list_path 文件是.lst格式，每行代表一张图片的具体信息，img_index \t label \t img_path
@click.option("-i", "list_path", required=True, help="list文件路径")
@click.option("-o", "output_path", required=True, help="lmdb输出路径")
@click.option("-l", "--log_level", default="INFO", help="日志输出登记,默认INFO")
@click.option("-b", "--batch_size", default=1000, help="每一次写入到lmdb的数量,默认1000")
@click.command()
def main(list_path: str, output_path: str, log_level: str, batch_size: int):
    import sys
    from loguru import logger
    exporter = IdtLmdbDataExporter(list_path, output_path, shape=(256, 256), batch_size=batch_size)
    logger.configure(**{"handlers": [
        {
            "sink": sys.stdout,
            "level": log_level.upper(),
        },
    ]})
    exporter.export()


if __name__ == "__main__":
    main()
