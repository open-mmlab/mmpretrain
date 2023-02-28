# flake8: noqa
import argparse
import re
import warnings
from collections import defaultdict
from pathlib import Path

from modelindex.load_model_index import load
from modelindex.models.ModelIndex import ModelIndex
from tabulate import tabulate

MMPT_ROOT = Path(__file__).absolute().parents[1]

prog_description = """\
Use metafile to generate a README.md.

Notice that the tool may fail in some corner cases, and you still need to check and fill some contents manually in the generated README.
"""

PREDICT_TEMPLATE = """\
**Predict image**

```python
from mmpretrain import inference_model

predict = inference_model('{model_name}', 'demo/bird.JPEG')
print(predict['pred_class'])
print(predict['pred_score'])
```
"""

RETRIEVE_TEMPLATE = """\
**Retrieve image**

```python
from mmpretrain import ImageRetrievalInferencer

inferencer = ImageRetrievalInferencer('{model_name}', prototype='demo/')
predict = inferencer('demo/dog.jpg', topk=2)[0]
print(predict[0])
print(predict[1])
```
"""

USAGE_TEMPLATE = """\
**Use the model**

```python
import torch
from mmpretrain import get_model

model = get_model('{model_name}', pretrained=True)
inputs = torch.rand(1, 3, 224, 224)
out = model(inputs)
print(type(out))
# To extract features.
feats = model.extract_feat(inputs)
print(type(feats))
```
"""

TRAIN_TEST_TEMPLATE = """\
**Train/Test Command**

Prepare your dataset according to the [docs](https://mmclassification.readthedocs.io/en/1.x/user_guides/dataset_prepare.html#prepare-dataset).

Train:

```shell
python tools/train.py {train_config}
```

Test:

```shell
python tools/test.py {test_config} {test_weights}
```
"""

TEST_ONLY_TEMPLATE = """\
**Test Command**

Prepare your dataset according to the [docs](https://mmclassification.readthedocs.io/en/1.x/user_guides/dataset_prepare.html#prepare-dataset).

Test:

```shell
python tools/test.py {test_config} {test_weights}
```
"""

METRIC_MAPPING = {
    'Top 1 Accuracy': 'Top-1 (%)',
    'Top 5 Accuracy': 'Top-5 (%)',
}

DATASET_PRIORITY = {
    'ImageNet-1k': 0,
    'CIFAR-10': 10,
    'CIFAR-100': 20,
}


def parse_args():
    parser = argparse.ArgumentParser(description=prog_description)
    parser.add_argument('metafile', type=Path, help='The path of metafile')
    parser.add_argument(
        '--table', action='store_true', help='Only generate summary tables')
    args = parser.parse_args()
    return args


def add_title(metafile: ModelIndex, readme: list):
    paper = metafile.collections[0].paper
    title = paper['Title']
    url = paper['URL']
    abbr = metafile.collections[0].name
    papertype = metafile.collections[0].data.get('type', 'Algorithm')

    readme.append(f'# {abbr}\n')
    readme.append(f'> [{title}]({url})\n')
    readme.append(f'<!-- [{papertype.upper()}] -->\n')


def add_abstract(metafile, readme):
    paper = metafile.collections[0].paper
    url = paper['URL']
    if 'arxiv' in url:
        try:
            import arxiv
            search = arxiv.Search(id_list=[url.split('/')[-1]])
            info = next(search.results())
            abstract = info.summary
        except ImportError:
            warnings.warn('Install arxiv parser by `pip install arxiv` '
                          'to automatically generate abstract.')
            abstract = None
    else:
        abstract = None

    readme.append('## Abstract\n')
    if abstract is not None:
        readme.append(abstract.replace('\n', ' '))

    readme.append('')
    readme.append('<div align=center>\n'
                  '<img src="" width="50%"/>\n'
                  '</div>')
    readme.append('')


def filter_models_by_task(models, task):
    model_list = []
    for model in models:
        if model.results is None and task is None:
            model_list.append(model)
        elif model.results is None:
            continue
        elif model.results[0].task == task or task == 'any':
            model_list.append(model)
    return model_list


def add_usage(metafile, readme):
    models = metafile.models
    if len(models) == 0:
        return

    readme.append('## How to use it?\n\n<!-- [TABS-BEGIN] -->\n')

    # Predict image
    cls_models = filter_models_by_task(models, 'Image Classification')
    if cls_models:
        model_name = cls_models[0].name
        readme.append(PREDICT_TEMPLATE.format(model_name=model_name))

    # Retrieve image
    retrieval_models = filter_models_by_task(models, 'Image Retrieval')
    if retrieval_models:
        model_name = retrieval_models[0].name
        readme.append(RETRIEVE_TEMPLATE.format(model_name=model_name))

    # Use the model
    model_name = models[0].name
    readme.append(USAGE_TEMPLATE.format(model_name=model_name))

    # Train/Test Command
    inputs = {}
    train_model = [
        model for model in models
        if 'headless' not in model.name and '3rdparty' not in model.name
    ]
    if train_model:
        template = TRAIN_TEST_TEMPLATE
        inputs['train_config'] = train_model[0].config
    else:
        template = TEST_ONLY_TEMPLATE
    test_model = filter_models_by_task(models, task='any')[0]
    inputs['test_config'] = test_model.config
    inputs['test_weights'] = test_model.weights
    readme.append(template.format(**inputs))

    readme.append('\n<!-- [TABS-END] -->\n')


def format_pretrain(pretrain_field):
    pretrain_infos = pretrain_field.split('-')[:-1]
    infos = []
    for info in pretrain_infos:
        if re.match('^\d+e$', info):
            info = f'{info[:-1]}-Epochs'
        elif re.match('^in\d+k$', info):
            info = f'ImageNet-{info[2:-1]}k'
        else:
            info = info.upper()
        infos.append(info)
    return ' '.join(infos)


def generate_model_table(models,
                         folder,
                         with_pretrain=True,
                         with_metric=True,
                         pretrained_models=[]):
    header = ['Model']
    if with_pretrain:
        header.append('Pretrain')
    header.extend(['Params (M)', 'Flops (G)'])
    if with_metric:
        metrics = set()
        for model in models:
            metrics.update(model.results[0].metrics.keys())
        metrics = sorted(list(set(metrics)))
        for metric in metrics:
            header.append(METRIC_MAPPING.get(metric, metric))
    header.extend(['Config', 'Download'])

    rows = []
    for model in models:
        model_name = f'`{model.name}`'
        config = (MMPT_ROOT / model.config).relative_to(folder)
        if model.weights is not None:
            download = f'[model]({model.weights})'
        else:
            download = 'N/A'

        if 'Converted From' in model.data:
            model_name += '\*'
            converted_from = model.data['Converted From']
        elif model.weights is not None:
            log = re.sub(r'.pth$', '.json', model.weights)
            download += f' \| [log]({log})'

        row = [model_name]
        if with_pretrain:
            pretrain_field = [
                field for field in model.name.split('_')
                if field.endswith('-pre')
            ]
            if pretrain_field:
                pretrain = format_pretrain(pretrain_field[0])
                upstream = [
                    pretrain_model for pretrain_model in pretrained_models
                    if model.name in pretrain_model.data.get('Downstream', [])
                ]
                if upstream:
                    pretrain = f'[{pretrain}]({upstream[0].weights})'
            else:
                pretrain = 'From scratch'
            row.append(pretrain)

        if model.metadata.parameters is not None:
            row.append(f'{model.metadata.parameters / 1e6:.2f}')  # Params
        else:
            row.append('N/A')
        if model.metadata.flops is not None:
            row.append(f'{model.metadata.flops / 1e9:.2f}')  # Params
        else:
            row.append('N/A')

        if with_metric:
            for metric in metrics:
                row.append(model.results[0].metrics.get(metric, 'N/A'))
        row.append(f'[config]({config})')
        row.append(download)

        rows.append(row)

    table_cfg = dict(
        tablefmt='pipe',
        floatfmt='.2f',
        colalign=['left'] + ['center'] * (len(row) - 1))
    table_string = tabulate(rows, header, **table_cfg)
    if any('Converted From' in model.data for model in models):
        table_string += (
            f"\n\n*Models with \* are converted from the [official repo]({converted_from['Code']}). "
            "The config files of these models are only for inference. We haven't reprodcue the training results.*\n"
        )

    return table_string + '\n'


def add_models(metafile, readme):
    models = metafile.models
    if len(models) == 0:
        return

    readme.append('## Models and results\n')
    algo_folder = Path(metafile.filepath).parent.absolute()

    # Pretrained models
    pretrain_models = filter_models_by_task(models, task=None)
    if pretrain_models:
        readme.append('### Pretrained models\n')
        readme.append(
            generate_model_table(
                pretrain_models,
                algo_folder,
                with_pretrain=False,
                with_metric=False))

    # Classification models
    tasks = [
        'Image Classification',
        'Image Retrieval',
        'Multi-Label Classification',
    ]

    for task in tasks:
        task_models = filter_models_by_task(models, task=task)
        if task_models:
            datasets = {model.results[0].dataset for model in task_models}
            datasets = sorted(
                list(datasets), key=lambda x: DATASET_PRIORITY.get(x, 50))
            for dataset in datasets:
                readme.append(f'### {task} on {dataset}\n')
                dataset_models = [
                    model for model in task_models
                    if model.results[0].dataset == dataset
                ]
                readme.append(
                    generate_model_table(
                        dataset_models,
                        algo_folder,
                        pretrained_models=pretrain_models))


def main():
    args = parse_args()
    metafile = load(str(args.metafile))
    readme_lines = []
    if not args.table:
        add_title(metafile, readme_lines)
        add_abstract(metafile, readme_lines)
        add_usage(metafile, readme_lines)
    add_models(metafile, readme_lines)
    if not args.table:
        readme_lines.append('## Citation\n')
        readme_lines.append('```bibtex\n```')
    print('\n'.join(readme_lines))


if __name__ == '__main__':
    main()
