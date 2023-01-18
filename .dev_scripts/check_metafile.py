import argparse
from pathlib import Path

import yaml
from modelindex.load_model_index import load
from modelindex.models.Collection import Collection
from modelindex.models.Model import Model
from modelindex.models.ModelIndex import ModelIndex

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
    args = parser.parse_args()
    return args


def check_collection(modelindex: ModelIndex):
    if len(modelindex.collections) != 1:
        return 'One metafile should have only one collection.'
    collection: Collection = modelindex.collections[0]
    if collection.name is None:
        return 'The collection should have `Name` field.'
    if collection.readme is None:
        return 'The collection should have `README` field.'
    if not (MMCLS_ROOT / collection.readme).exists():
        return f'The README {collection.readme} is not found.'
    if not isinstance(collection.paper, dict):
        return ('The collection should have `Paper` field with '
                '`Title` and `URL`.')
    if 'Title' not in collection.paper:
        # URL is not necessary.
        return "The collection's paper should have `Paper` field."


def check_model(model: Model, wall=True):
    if model.name is None:
        return "A model doesn't have `Name` field."
    if model.metadata is None:
        return f'{model.name}: No `Metadata` field.'
    if model.metadata.parameters is None or model.metadata.flops is None:
        return (
            f'{model.name}: Metadata should have `Parameters` and '
            '`FLOPs` fields. You can use `tools/analysis_tools/get_flops.py` '
            'to calculate them.')
    if model.results is not None:
        result = model.results[0]
        if not isinstance(result.dataset, str):
            return (
                f'{model.name}: Dataset field of Results should be a string. '
                'If you want to specify the training dataset, please use '
                '`Metadata.Training Data` field.')
    if model.config is None:
        return f'{model.name}: No `Config` field.'
    if not (MMCLS_ROOT / model.config).exists():
        return f'{model.name}: The config {model.config} is not found.'
    if model.in_collection is None:
        return f'{model.name}: No `In Collection` field.'

    if wall and model.data.get(
            'Converted From') is not None and '3rdparty' not in model.name:
        print(f'WARN: The model name {model.name} should include '
              "'3rdparty' since it's converted from other repository.")
    if wall and model.weights is not None and model.weights.endswith('.pth'):
        basename = model.weights.rsplit('/', 1)[-1]
        if not basename.startswith(model.name):
            print(f'WARN: The checkpoint name {basename} is not the '
                  f'same as the model name {model.name}.')


def main(metafile: Path, args):
    if metafile.name != 'metafile.yml':
        # Avoid checking other yaml file.
        return
    elif metafile.samefile(MMCLS_ROOT / 'model-index.yml'):
        return

    with open(MMCLS_ROOT / 'model-index.yml', 'r') as f:
        metafile_list = yaml.load(f, yaml.Loader)['Import']
        if not any(
                metafile.samefile(MMCLS_ROOT / file)
                for file in metafile_list):
            raise ValueError(f'The metafile {metafile} is not imported in '
                             'the `model-index.yml`.')

    modelindex = load(str(metafile))
    modelindex.build_models_with_collections()
    collection_err = check_collection(modelindex)
    if collection_err is not None:
        raise ValueError(f'The `Collections` in the {metafile} is wrong:'
                         f'\n\t{collection_err}')
    for model in modelindex.models:
        model_err = check_model(model, args.Wall)
        if model_err is not None:
            raise ValueError(
                f'The `Models` in the {metafile} is wrong:\n\t{model_err}')


if __name__ == '__main__':
    args = parse_args()
    for metafile in args.metafile:
        main(metafile, args)
