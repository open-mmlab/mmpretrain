import argparse
import logging
import re
import sys
from pathlib import Path

import yaml
from modelindex.load_model_index import load
from modelindex.models.Collection import Collection
from modelindex.models.Model import Model
from modelindex.models.ModelIndex import ModelIndex


class ContextFilter(logging.Filter):
    metafile = None
    name = None
    failed = False

    def filter(self, record: logging.LogRecord):
        record.color = {
            logging.WARNING: '\x1b[33;20m',
            logging.ERROR: '\x1b[31;1m',
        }.get(record.levelno, '')
        self.failed = self.failed or (record.levelno >= logging.ERROR)
        record.metafile = self.metafile or ''
        record.name = ('' if self.name is None else '\x1b[32m' + self.name +
                       '\x1b[0m: ')
        return True


context = ContextFilter()
logging.basicConfig(
    format='[%(metafile)s] %(color)s%(levelname)s\x1b[0m - %(name)s%(message)s'
)
logger = logging.getLogger()
logger.addFilter(context)

prog_description = """\
Check the format of metafile.
"""

MMCLS_ROOT = Path(__file__).absolute().parents[1]


def parse_args():
    parser = argparse.ArgumentParser(description=prog_description)
    parser.add_argument(
        'metafile', type=Path, nargs='+', help='The path of the matafile.')
    parser.add_argument(
        '--Wall',
        '-w',
        action='store_true',
        help='Whether to enable all warnings.')
    parser.add_argument('--skip', action='append', help='Rules to skip check.')
    args = parser.parse_args()
    args.skip = args.skip or []
    return args


def check_collection(modelindex: ModelIndex, skip=[]):

    if len(modelindex.collections) == 0:
        return ['No collection field.']
    elif len(modelindex.collections) > 1:
        logger.error('One metafile should have only one collection.')

    collection: Collection = modelindex.collections[0]

    if collection.name is None:
        logger.error('The collection should have `Name` field.')
    if collection.readme is None:
        logger.error('The collection should have `README` field.')
    if not (MMCLS_ROOT / collection.readme).exists():
        logger.error(f'The README {collection.readme} is not found.')
    if not isinstance(collection.paper, dict):
        logger.error('The collection should have `Paper` field with '
                     '`Title` and `URL`.')
    elif 'Title' not in collection.paper:
        # URL is not necessary.
        logger.error("The collection's paper should have `Paper` field.")


def check_model_name(name):
    fields = name.split('_')

    if len(fields) > 5:
        logger.warning('Too many fields.')
        return
    elif len(fields) < 3:
        logger.warning('Too few fields.')
        return
    elif len(fields) == 5:
        algo, model, pre, train, data = fields
    elif len(fields) == 3:
        model, train, data = fields
        algo, pre = None, None
    elif len(fields) == 4 and fields[1].endswith('-pre'):
        model, pre, train, data = fields
        algo = None
    else:
        algo, model, train, data = fields
        pre = None

    if pre is not None and not pre.endswith('-pre'):
        logger.warning(f'The position of `{pre}` should be '
                       'pre-training information, and ends with `-pre`.')

    if '3rdparty' not in train and re.match(r'\d+xb\d+', train) is None:
        logger.warning(f'The position of `{train}` should be training '
                       'infomation, and starts with `3rdparty` or '
                       '`{num_device}xb{batch_per_device}`')


def check_model(model: Model, skip=[]):

    context.name = None
    if model.name is None:
        logger.error("A model doesn't have `Name` field.")
        return
    context.name = model.name
    check_model_name(model.name)

    if model.name.endswith('.py'):
        logger.error("Don't add `.py` suffix in model name.")

    if model.metadata is None and 'metadata' not in skip:
        logger.error('No `Metadata` field.')

    if (model.metadata.parameters is None
            or model.metadata.flops is None) and 'flops-param' not in skip:
        logger.error('Metadata should have `Parameters` and `FLOPs` fields. '
                     'You can use `tools/analysis_tools/get_flops.py` '
                     'to calculate them.')

    if model.results is not None and 'result' not in skip:
        result = model.results[0]
        if not isinstance(result.dataset, str):
            logger.error('Dataset field of Results should be a string. '
                         'If you want to specify the training dataset, '
                         'please use `Metadata.Training Data` field.')

    if 'config' not in skip:
        if model.config is None:
            logger.error('No `Config` field.')
        elif not (MMCLS_ROOT / model.config).exists():
            logger.error(f'The config {model.config} is not found.')

    if model.in_collection is None:
        logger.error('No `In Collection` field.')

    if (model.data.get('Converted From') is not None
            and '3rdparty' not in model.name):
        logger.warning("The model name should include '3rdparty' "
                       "since it's converted from other repository.")

    if (model.weights is not None and model.weights.endswith('.pth')
            and 'ckpt-name' not in skip):
        basename = model.weights.rsplit('/', 1)[-1]
        if not basename.startswith(model.name):
            logger.warning(f'The checkpoint name {basename} is not the '
                           'same as the model name.')

    context.name = None


def main(metafile: Path, args):
    if metafile.name != 'metafile.yml':
        # Avoid checking other yaml file.
        return
    elif metafile.samefile(MMCLS_ROOT / 'model-index.yml'):
        return

    context.metafile = metafile

    with open(MMCLS_ROOT / 'model-index.yml', 'r') as f:
        metafile_list = yaml.load(f, yaml.Loader)['Import']
        if not any(
                metafile.samefile(MMCLS_ROOT / file)
                for file in metafile_list):
            logger.error(
                'The metafile is not imported in the `model-index.yml`.')

    modelindex = load(str(metafile))
    modelindex.build_models_with_collections()
    check_collection(modelindex, args.skip)

    names = {model.name for model in modelindex.models}

    for model in modelindex.models:
        check_model(model, args.skip)

        for downstream in model.data.get('Downstream', []):
            if downstream not in names:
                context.name = model.name
                logger.error(
                    f"The downstream model {downstream} doesn't exist.")


if __name__ == '__main__':
    args = parse_args()
    if args.Wall:
        logger.setLevel(logging.WARNING)
    else:
        logger.setLevel(logging.ERROR)
    for metafile in args.metafile:
        main(metafile, args)
    sys.exit(int(context.failed))
