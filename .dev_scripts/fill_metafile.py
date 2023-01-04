import argparse
import copy
import pkg_resources
from functools import partial
from pathlib import Path

import yaml
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm, Prompt
from rich.syntax import Syntax

from mmcls import digit_version

if digit_version(pkg_resources.get_distribution(
        'rich').version) < digit_version('12.0'):
    # The rich is not compatible with readline
    # see https://github.com/Textualize/rich/issues/2293
    # To use readline, you need `pip install rich<12.0`
    import readline
    readline.set_completer_delims('\t')
    readline.parse_and_bind('tab: complete')

prog_description = """\
To display metafile or fill missing fields of the metafile.
"""

MMCLS_ROOT = Path(__file__).absolute().parents[1].resolve().absolute()
console = Console()


class MyDumper(yaml.Dumper):

    def increase_indent(self, flow=False, indentless=False):
        return super(MyDumper, self).increase_indent(flow, False)


yaml_dump = partial(
    yaml.dump, Dumper=MyDumper, default_flow_style=False, sort_keys=False)


def parse_args():
    parser = argparse.ArgumentParser(description=prog_description)
    parser.add_argument('--src', type=Path, help='The path of the matafile.')
    parser.add_argument('--out', '-o', type=Path, help='The output path.')
    parser.add_argument(
        '--view', action='store_true', help='Only pretty print the metafile.')
    args = parser.parse_args()
    return args


def get_flops(config_path):
    import numpy as np
    import torch
    from fvcore.nn import FlopCountAnalysis, parameter_count
    from mmengine.config import Config
    from mmengine.dataset import Compose
    from mmengine.registry import DefaultScope

    import mmcls.datasets  # noqa: F401
    from mmcls.apis import init_model

    cfg = Config.fromfile(config_path)

    if 'test_dataloader' in cfg:
        # build the data pipeline
        test_dataset = cfg.test_dataloader.dataset
        if test_dataset.pipeline[0]['type'] == 'LoadImageFromFile':
            test_dataset.pipeline.pop(0)
        if test_dataset.type in ['CIFAR10', 'CIFAR100']:
            # The image shape of CIFAR is (32, 32, 3)
            test_dataset.pipeline.insert(1, dict(type='Resize', scale=32))

        with DefaultScope.overwrite_default_scope('mmcls'):
            data = Compose(test_dataset.pipeline)({
                'img':
                np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
            })
        resolution = tuple(data['inputs'].shape[-2:])
    else:
        # For configs only for get model.
        resolution = (224, 224)

    model = init_model(cfg, device='cpu')

    with torch.no_grad():
        model.forward = model.extract_feat
        model.to('cpu')
        inputs = (torch.randn((1, 3, *resolution)), )
        flops = FlopCountAnalysis(model, inputs).total()
        params = parameter_count(model)['']
    return int(flops), int(params)


def fill_collection(collection: dict):
    if collection.get('Name') is None:
        name = Prompt.ask('Please input the collection [red]name[/]')
        while name == '':
            name = Prompt.ask('Please input the collection [red]name[/]')
        collection['Name'] = name

    if collection.get('Metadata', {}).get('Architecture') is None:
        architecture = []
        arch = Prompt.ask('Please input the model [red]architecture[/] '
                          '(input empty to finish)')
        while arch != '':
            architecture.append(arch)
            arch = Prompt.ask('Please input the model [red]architecture[/] '
                              '(input empty to finish)')
        if len(architecture) > 0:
            collection.setdefault('Metadata', {})
            collection['Metadata']['Architecture'] = architecture

    if collection.get('Paper', {}).get('Title') is None:
        title = Prompt.ask('Please input the [red]paper title[/]') or None
    else:
        title = collection['Paper']['Title']
    if collection.get('Paper', {}).get('URL') is None:
        url = Prompt.ask('Please input the [red]paper url[/]') or None
    else:
        url = collection['Paper']['URL']
    paper = dict(Title=title, URL=url)
    collection['Paper'] = paper

    if collection.get('README') is None:
        readme = Prompt.ask(
            'Please input the [red]README[/] file path') or None
        if readme is not None:
            collection['README'] = str(
                Path(readme).absolute().relative_to(MMCLS_ROOT))
        else:
            collection['README'] = None

    order = ['Name', 'Metadata', 'Paper', 'README', 'Code']
    collection = {
        k: collection[k]
        for k in sorted(collection.keys(), key=order.index)
    }
    return collection


def fill_model(model: dict, defaults: dict):
    if model.get('Name') is None:
        name = Prompt.ask('Please input the model [red]name[/]')
        while name == '':
            name = Prompt.ask('Please input the model [red]name[/]')
        model['Name'] = name

    model['In Collection'] = defaults.get('In Collection')

    config = model.get('Config')
    if config is None:
        config = Prompt.ask(
            'Please input the [red]config[/] file path') or None
        if config is not None:
            config = str(Path(config).absolute().relative_to(MMCLS_ROOT))
    model['Config'] = config

    if model.get('Metadata', {}).get('Training Data') is None:
        training_data = []
        dataset = Prompt.ask('Please input all [red]training dataset[/], '
                             'include pre-training (input empty to finish)')
        while dataset != '':
            training_data.append(dataset)
            dataset = Prompt.ask(
                'Please input all [red]training dataset[/], '
                'include pre-training (input empty to finish)')
        if len(training_data) > 1:
            model.setdefault('Metadata', {})
            model['Metadata']['Training Data'] = training_data
        elif len(training_data) == 1:
            model.setdefault('Metadata', {})
            model['Metadata']['Training Data'] = training_data[0]

    flops = model.get('Metadata', {}).get('FLOPs')
    params = model.get('Metadata', {}).get('Parameters')
    if flops is None:
        if model.get('Config') is None:
            flops = Prompt.ask('Please specify the [red]FLOPs[/] manually '
                               'since no config file.') or None
            if flops is not None:
                flops = int(flops)
    if params is None:
        if model.get('Config') is None:
            params = Prompt.ask(
                'Please specify the [red]number of parameters[/] '
                'manually since no config file') or None
            if params is not None:
                params = int(params)
    if model.get('Config') is not None and (
            MMCLS_ROOT / model['Config']).exists() and (flops is None
                                                        or params is None):
        flops, params = get_flops(str(MMCLS_ROOT / model['Config']))
    model.setdefault('Metadata', {})
    model['Metadata'].setdefault('FLOPs', flops)
    model['Metadata'].setdefault('Parameters', params)

    results = model.get('Results')
    if results is None:
        test_dataset = Prompt.ask(
            'Please input the [red]test dataset[/]') or None
        if test_dataset is not None:
            task = Prompt.ask(
                'Please input the [red]test task[/]',
                default='Image Classification')
            if task == 'Image Classification':
                metrics = {}
                top1 = Prompt.ask(
                    'Please input the [red]top-1 accuracy[/]') or None
                top5 = Prompt.ask(
                    'Please input the [red]top-5 accuracy[/]') or None
                if top1 is not None:
                    metrics['Top 1 Accuracy'] = round(float(top1), 2)
                if top5 is not None:
                    metrics['Top 5 Accuracy'] = round(float(top5), 2)
            else:
                metrics = {}
                metric = Prompt.ask(
                    'Please input the [red]metrics[/] like "mAP=94.98" '
                    '(input empty to finish)')
                while metric != '':
                    k, v = metric.split('=')[:2]
                    metrics[k] = round(float(v), 2)
                    metric = Prompt.ask(
                        'Please input the [red]metrics[/] like "mAP=94.98" '
                        '(input empty to finish)')
            if len(metrics) > 0:
                results = [{
                    'Dataset': test_dataset,
                    'Metrics': metrics,
                    'Task': task
                }]
    model['Results'] = results

    weights = model.get('Weights')
    if weights is None:
        weights = Prompt.ask(
            'Please input the [red]checkpoint download link[/]') or None
    model['Weights'] = weights

    if model.get('Converted From') is None and model.get(
            'Weights') is not None:
        if Confirm.ask(
                'Is the checkpoint is converted from [red]other repository[/]?'
        ):
            converted_from = {}
            converted_from['Weights'] = Prompt.ask(
                'Please fill the original checkpoint download link')
            converted_from['Code'] = Prompt.ask(
                'Please fill the original repository link',
                default=defaults.get('Convert From.Code', None))
            defaults['Convert From.Code'] = converted_from['Code']
            model['Converted From'] = converted_from
    else:
        defaults['Convert From.Code'] = model['Converted From']['Code']

    order = [
        'Name', 'Metadata', 'In Collection', 'Results', 'Weights', 'Config',
        'Converted From'
    ]
    model = {k: model[k] for k in sorted(model.keys(), key=order.index)}
    return model


def format_collection(collection: dict):
    yaml_str = yaml_dump(collection)
    return Panel(
        Syntax(yaml_str, 'yaml', background_color='default'),
        width=150,
        title='Collection')


def format_model(model: dict):
    yaml_str = yaml_dump(model)
    return Panel(
        Syntax(yaml_str, 'yaml', background_color='default'),
        width=150,
        title='Model')


def main():
    args = parse_args()

    if args.src is not None:
        with open(args.src, 'r') as f:
            content = yaml.load(f, yaml.SafeLoader)
    else:
        content = {}

    if args.view:
        collection = content.get('Collections', [{}])[0]
        console.print(format_collection(collection))
        models = content.get('Models', [])
        for model in models:
            console.print(format_model(model))
        return

    collection = content.get('Collections', [{}])[0]
    ori_collection = copy.deepcopy(collection)

    console.print(format_collection(collection))
    collection = fill_collection(collection)
    if ori_collection != collection:
        console.print(format_collection(collection))
    model_defaults = {'In Collection': collection['Name']}

    models = content.get('Models', [])
    updated_models = []

    try:
        for model in models:
            console.print(format_model(model))
            ori_model = copy.deepcopy(model)
            model = fill_model(model, model_defaults)
            if ori_model != model:
                console.print(format_model(model))
            updated_models.append(model)

        while Confirm.ask('Add new model?'):
            model = fill_model({}, model_defaults)
            updated_models.append(model)
    finally:
        # Save updated models even error happened.
        updated_models.sort(key=lambda item: (item.get('Metadata', {}).get(
            'FLOPs', 0), len(item['Name'])))
        if args.out is not None:
            with open(args.out, 'w') as f:
                yaml_dump({'Collections': [collection]}, f)
                f.write('\n')
                yaml_dump({'Models': updated_models}, f)
        else:
            modelindex = {
                'Collections': [collection],
                'Models': updated_models
            }
            yaml_str = yaml_dump(modelindex)
            console.print(Syntax(yaml_str, 'yaml', background_color='default'))
            console.print('Specify [red]`--out`[/] to dump to file.')


if __name__ == '__main__':
    main()