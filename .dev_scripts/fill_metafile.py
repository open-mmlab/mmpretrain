import argparse
import copy
import re
from functools import partial
from pathlib import Path

import yaml
from prompt_toolkit import ANSI
from prompt_toolkit import prompt as _prompt
from prompt_toolkit.completion import (FuzzyCompleter, FuzzyWordCompleter,
                                       PathCompleter)
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm, Prompt
from rich.syntax import Syntax

prog_description = """\
To display metafile or fill missing fields of the metafile.
"""

MMCLS_ROOT = Path(__file__).absolute().parents[1].resolve().absolute()
console = Console()
dataset_completer = FuzzyWordCompleter([
    'ImageNet-1k', 'ImageNet-21k', 'CIFAR-10', 'CIFAR-100', 'RefCOCO', 'VQAv2',
    'COCO', 'OpenImages', 'Object365', 'CC3M', 'CC12M', 'YFCC100M', 'VG'
])


def prompt(message,
           allow_empty=True,
           default=None,
           multiple=False,
           completer=None):
    with console.capture() as capture:
        console.print(message, end='')

    message = ANSI(capture.get())
    ask = partial(
        _prompt, message=message, default=default or '', completer=completer)

    out = ask()

    if multiple:
        outs = []
        while out != '':
            outs.append(out)
            out = ask()
        return outs

    if not allow_empty and out == '':
        while out == '':
            out = ask()

    if default is None and out == '':
        return None
    else:
        return out.strip()


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
        '--inplace',
        '-i',
        action='store_true',
        help='Modify the source metafile inplace.')
    parser.add_argument(
        '--view', action='store_true', help='Only pretty print the metafile.')
    parser.add_argument('--csv', type=str, help='Use a csv to update models.')
    args = parser.parse_args()
    if args.inplace:
        args.out = args.src
    return args


def get_flops_params(config_path):
    import numpy as np
    import torch
    from mmengine.analysis import FlopAnalyzer, parameter_count
    from mmengine.dataset import Compose
    from mmengine.model.utils import revert_sync_batchnorm
    from mmengine.registry import DefaultScope

    from mmpretrain.apis import get_model
    from mmpretrain.models.utils import no_load_hf_pretrained_model

    with no_load_hf_pretrained_model():
        model = get_model(config_path, device='cpu')
    model = revert_sync_batchnorm(model)
    model.eval()
    params = int(parameter_count(model)[''])

    # get flops
    try:
        if 'test_dataloader' in model._config:
            # build the data pipeline
            test_dataset = model._config.test_dataloader.dataset
            if test_dataset.pipeline[0]['type'] == 'LoadImageFromFile':
                test_dataset.pipeline.pop(0)
            if test_dataset.type in ['CIFAR10', 'CIFAR100']:
                # The image shape of CIFAR is (32, 32, 3)
                test_dataset.pipeline.insert(1, dict(type='Resize', scale=32))

            with DefaultScope.overwrite_default_scope('mmpretrain'):
                data = Compose(test_dataset.pipeline)({
                    'img':
                    np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
                })
            resolution = tuple(data['inputs'].shape[-2:])
        else:
            # For configs only for get model.
            resolution = (224, 224)

        with torch.no_grad():
            # Skip flops if the model doesn't have `extract_feat` method.
            model.forward = model.extract_feat
            model.to('cpu')
            inputs = (torch.randn((1, 3, *resolution)), )
            analyzer = FlopAnalyzer(model, inputs)
            analyzer.unsupported_ops_warnings(False)
            analyzer.uncalled_modules_warnings(False)
            flops = int(analyzer.total())
    except Exception:
        print('Unable to calculate flops.')
        flops = None
    return flops, params


def fill_collection(collection: dict):
    if collection.get('Name') is None:
        name = prompt(
            'Please input the collection [red]name[/]: ', allow_empty=False)
        collection['Name'] = name

    if collection.get('Metadata', {}).get('Architecture') is None:
        architecture = prompt(
            'Please input the model [red]architecture[/] '
            '(input empty to finish): ',
            multiple=True)
        if len(architecture) > 0:
            collection.setdefault('Metadata', {})
            collection['Metadata']['Architecture'] = architecture

    if collection.get('Paper', {}).get('Title') is None:
        title = prompt('Please input the [red]paper title[/]: ')
    else:
        title = collection['Paper']['Title']
    if collection.get('Paper', {}).get('URL') is None:
        url = prompt('Please input the [red]paper url[/]: ')
    else:
        url = collection['Paper']['URL']
    paper = dict(Title=title, URL=url)
    collection['Paper'] = paper

    if collection.get('README') is None:
        readme = prompt(
            'Please input the [red]README[/] file path: ',
            completer=PathCompleter(file_filter=lambda name: Path(name).is_dir(
            ) or 'README.md' in name))
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


def fill_model_by_prompt(model: dict, defaults: dict):
    # Name
    if model.get('Name') is None:
        name = prompt(
            'Please input the model [red]name[/]: ', allow_empty=False)
        model['Name'] = name

    # In Collection
    model['In Collection'] = defaults.get('In Collection')

    # Config
    config = model.get('Config')
    if config is None:
        config = prompt(
            'Please input the [red]config[/] file path: ',
            completer=FuzzyCompleter(PathCompleter()))
        if config is not None:
            config = str(Path(config).absolute().relative_to(MMCLS_ROOT))
    model['Config'] = config

    # Metadata.Flops, Metadata.Parameters
    flops = model.get('Metadata', {}).get('FLOPs')
    params = model.get('Metadata', {}).get('Parameters')
    if model.get('Config') is not None and (
            MMCLS_ROOT / model['Config']).exists() and (flops is None
                                                        and params is None):
        print('Automatically compute FLOPs and Parameters from config.')
        flops, params = get_flops_params(str(MMCLS_ROOT / model['Config']))

    if flops is None:
        flops = prompt('Please specify the [red]FLOPs[/]: ')
        if flops is not None:
            flops = int(flops)
    if params is None:
        params = prompt('Please specify the [red]number of parameters[/]: ')
        if params is not None:
            params = int(params)

    model.setdefault('Metadata', {})
    model['Metadata'].setdefault('FLOPs', flops)
    model['Metadata'].setdefault('Parameters', params)

    if 'Training Data' not in model.get('Metadata', {}) and \
            'Training Data' not in defaults.get('Metadata', {}):
        training_data = prompt(
            'Please input all [red]training dataset[/], '
            'include pre-training (input empty to finish): ',
            completer=dataset_completer,
            multiple=True)
        if len(training_data) > 1:
            model['Metadata']['Training Data'] = training_data
        elif len(training_data) == 1:
            model['Metadata']['Training Data'] = training_data[0]

    results = model.get('Results')
    if results is None:
        test_dataset = prompt(
            'Please input the [red]test dataset[/]: ',
            completer=dataset_completer)
        if test_dataset is not None:
            task = Prompt.ask(
                'Please input the [red]test task[/]',
                default='Image Classification')
            if task == 'Image Classification':
                metrics = {}
                top1 = prompt('Please input the [red]top-1 accuracy[/]: ')
                top5 = prompt('Please input the [red]top-5 accuracy[/]: ')
                if top1 is not None:
                    metrics['Top 1 Accuracy'] = round(float(top1), 2)
                if top5 is not None:
                    metrics['Top 5 Accuracy'] = round(float(top5), 2)
            else:
                metrics_list = prompt(
                    'Please input the [red]metrics[/] like "mAP=94.98" '
                    '(input empty to finish): ',
                    multiple=True)
                metrics = {}
                for metric in metrics_list:
                    k, v = metric.split('=')[:2]
                    metrics[k] = round(float(v), 2)
            results = [{
                'Task': task,
                'Dataset': test_dataset,
                'Metrics': metrics or None,
            }]
    model['Results'] = results

    weights = model.get('Weights')
    if weights is None:
        weights = prompt('Please input the [red]checkpoint download link[/]: ')
    model['Weights'] = weights

    if model.get('Converted From') is None and model.get(
            'Weights') is not None:
        if '3rdparty' in model['Name'] or Confirm.ask(
                'Is the checkpoint is converted '
                'from [red]other repository[/]?',
                default=False):
            converted_from = {}
            converted_from['Weights'] = prompt(
                'Please fill the original checkpoint download link: ')
            converted_from['Code'] = Prompt.ask(
                'Please fill the original repository link',
                default=defaults.get('Convert From.Code', None))
            defaults['Convert From.Code'] = converted_from['Code']
            model['Converted From'] = converted_from
    elif model.get('Converted From', {}).get('Code') is not None:
        defaults['Convert From.Code'] = model['Converted From']['Code']

    order = [
        'Name', 'Metadata', 'In Collection', 'Results', 'Weights', 'Config',
        'Converted From', 'Downstream'
    ]
    model = {k: model[k] for k in sorted(model.keys(), key=order.index)}
    return model


def update_model_by_dict(model: dict, update_dict: dict, defaults: dict):
    # Name
    if 'name override' in update_dict:
        model['Name'] = update_dict['name override'].strip()

    # In Collection
    model['In Collection'] = defaults.get('In Collection')

    # Config
    if 'config' in update_dict:
        config = update_dict['config'].strip()
        config = str(Path(config).absolute().relative_to(MMCLS_ROOT))
        config_updated = (config != model.get('Config'))
        model['Config'] = config
    else:
        config_updated = False

    # Metadata.Flops, Metadata.Parameters
    flops = model.get('Metadata', {}).get('FLOPs')
    params = model.get('Metadata', {}).get('Parameters')
    if config_updated or (flops is None and params is None):
        print(f'Automatically compute FLOPs and Parameters of {model["Name"]}')
        flops, params = get_flops_params(str(MMCLS_ROOT / model['Config']))

    model.setdefault('Metadata', {})
    model['Metadata']['FLOPs'] = flops
    model['Metadata']['Parameters'] = params

    # Metadata.Training Data
    if 'training dataset' in update_dict:
        train_data = update_dict['training dataset'].strip()
        train_data = re.split(r'\s+', train_data)
        if len(train_data) > 1:
            model['Metadata']['Training Data'] = train_data
        elif len(train_data) == 1:
            model['Metadata']['Training Data'] = train_data[0]

    # Results.Dataset
    if 'test dataset' in update_dict:
        test_data = update_dict['test dataset'].strip()
        results = model.get('Results') or [{}]
        result = results[0]
        result['Dataset'] = test_data
        model['Results'] = results

    # Results.Metrics.Top 1 Accuracy
    result = None
    if 'top-1' in update_dict:
        top1 = update_dict['top-1']
        results = model.get('Results') or [{}]
        result = results[0]
        result.setdefault('Metrics', {})
        result['Metrics']['Top 1 Accuracy'] = round(float(top1), 2)
        task = 'Image Classification'
        model['Results'] = results

    # Results.Metrics.Top 5 Accuracy
    if 'top-5' in update_dict:
        top5 = update_dict['top-5']
        results = model.get('Results') or [{}]
        result = results[0]
        result.setdefault('Metrics', {})
        result['Metrics']['Top 5 Accuracy'] = round(float(top5), 2)
        task = 'Image Classification'
        model['Results'] = results

    if result is not None:
        result['Metrics']['Task'] = task

    # Weights
    if 'weights' in update_dict:
        weights = update_dict['weights'].strip()
        model['Weights'] = weights

    # Converted From.Code
    if 'converted from.code' in update_dict:
        from_code = update_dict['converted from.code'].strip()
        model.setdefault('Converted From', {})
        model['Converted From']['Code'] = from_code

    # Converted From.Weights
    if 'converted from.weights' in update_dict:
        from_weight = update_dict['converted from.weights'].strip()
        model.setdefault('Converted From', {})
        model['Converted From']['Weights'] = from_weight

    order = [
        'Name', 'Metadata', 'In Collection', 'Results', 'Weights', 'Config',
        'Converted From', 'Downstream'
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


def order_models(model):
    order = []
    # Pre-trained model
    order.append(int('Downstream' not in model))
    # non-3rdparty model
    order.append(int('3rdparty' in model['Name']))
    # smaller model
    order.append(model.get('Metadata', {}).get('Parameters', 0))
    # faster model
    order.append(model.get('Metadata', {}).get('FLOPs', 0))
    # name order
    order.append(len(model['Name']))

    return tuple(order)


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
    model_defaults = {
        'In Collection': collection['Name'],
        'Metadata': collection.get('Metadata', {}),
    }

    models = content.get('Models', [])
    updated_models = []

    if args.csv is not None:
        import pandas as pd
        df = pd.read_csv(args.csv).rename(columns=lambda x: x.strip().lower())
        assert df['name'].is_unique, 'The csv has duplicated model names.'
        models_dict = {item['Name']: item for item in models}
        for update_dict in df.to_dict('records'):
            assert 'name' in update_dict, 'The csv must have the `Name` field.'
            model_name = update_dict['name'].strip()
            model = models_dict.pop(model_name, {'Name': model_name})
            model = update_model_by_dict(model, update_dict, model_defaults)
            updated_models.append(model)
        updated_models.extend(models_dict.values())
    else:
        for model in models:
            console.print(format_model(model))
            ori_model = copy.deepcopy(model)
            model = fill_model_by_prompt(model, model_defaults)
            if ori_model != model:
                console.print(format_model(model))
            updated_models.append(model)

        while Confirm.ask('Add new model?', default=False):
            model = fill_model_by_prompt({}, model_defaults)
            updated_models.append(model)

    # Save updated models even error happened.
    updated_models.sort(key=order_models)
    if args.out is not None:
        with open(args.out, 'w') as f:
            yaml_dump({'Collections': [collection]}, f)
            f.write('\n')
            yaml_dump({'Models': updated_models}, f)
    else:
        modelindex = {'Collections': [collection], 'Models': updated_models}
        yaml_str = yaml_dump(modelindex)
        console.print(Syntax(yaml_str, 'yaml', background_color='default'))
        console.print('Specify [red]`--out`[/] to dump to file.')


if __name__ == '__main__':
    main()
