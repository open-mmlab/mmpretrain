# flake8: noqa
import argparse
import re
import warnings
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

Prepare your dataset according to the [docs](https://mmpretrain.readthedocs.io/en/latest/user_guides/dataset_prepare.html#prepare-dataset).

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

Prepare your dataset according to the [docs](https://mmpretrain.readthedocs.io/en/latest/user_guides/dataset_prepare.html#prepare-dataset).

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
    parser.add_argument(
        '--update', type=str, help='Update the specified readme file.')
    parser.add_argument('--out', type=str, help='Output to the file.')
    parser.add_argument(
        '--update-items',
        type=str,
        nargs='+',
        default=['models'],
        help='Update the specified readme file.')
    args = parser.parse_args()
    return args


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


def add_title(metafile: ModelIndex):
    paper = metafile.collections[0].paper
    title = paper['Title']
    url = paper['URL']
    abbr = metafile.collections[0].name
    papertype = metafile.collections[0].data.get('type', 'Algorithm')

    return f'# {abbr}\n> [{title}]({url})\n<!-- [{papertype.upper()}] -->\n'


def add_abstract(metafile: ModelIndex):
    paper = metafile.collections[0].paper
    url = paper['URL']
    if 'arxiv' in url:
        try:
            import arxiv
            search = arxiv.Search(id_list=[url.split('/')[-1]])
            info = next(search.results())
            abstract = info.summary.replace('\n', ' ')
        except ImportError:
            warnings.warn('Install arxiv parser by `pip install arxiv` '
                          'to automatically generate abstract.')
            abstract = None
    else:
        abstract = None

    content = '## Abstract\n'
    if abstract is not None:
        content += f'\n{abstract}\n'
    return content


def add_usage(metafile):
    models = metafile.models
    if len(models) == 0:
        return

    content = []
    content.append('## How to use it?\n\n<!-- [TABS-BEGIN] -->\n')

    # Predict image
    cls_models = filter_models_by_task(models, 'Image Classification')
    if cls_models:
        model_name = cls_models[0].name
        content.append(PREDICT_TEMPLATE.format(model_name=model_name))

    # Retrieve image
    retrieval_models = filter_models_by_task(models, 'Image Retrieval')
    if retrieval_models:
        model_name = retrieval_models[0].name
        content.append(RETRIEVE_TEMPLATE.format(model_name=model_name))

    # Use the model
    model_name = models[0].name
    content.append(USAGE_TEMPLATE.format(model_name=model_name))

    # Train/Test Command
    inputs = {}
    train_model = [
        model for model in models
        if 'headless' not in model.name and '3rdparty' not in model.name
    ]
    if train_model:
        template = TRAIN_TEST_TEMPLATE
        inputs['train_config'] = train_model[0].config
    elif len(filter_models_by_task(models, task='any')) > 0:
        template = TEST_ONLY_TEMPLATE
    else:
        content.append('\n<!-- [TABS-END] -->\n')
        return '\n'.join(content)

    test_model = filter_models_by_task(models, task='any')[0]
    inputs['test_config'] = test_model.config
    inputs['test_weights'] = test_model.weights
    content.append(template.format(**inputs))

    content.append('\n<!-- [TABS-END] -->\n')
    return '\n'.join(content)


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
    table_string = tabulate(rows, header, **table_cfg) + '\n'
    if any('Converted From' in model.data for model in models):
        table_string += (
            f"\n*Models with \* are converted from the [official repo]({converted_from['Code']}). "
            "The config files of these models are only for inference. We haven't reproduce the training results.*\n"
        )

    return table_string


def add_models(metafile):
    models = metafile.models
    if len(models) == 0:
        return ''

    content = ['## Models and results\n']
    algo_folder = Path(metafile.filepath).parent.absolute().resolve()

    # Pretrained models
    pretrain_models = filter_models_by_task(models, task=None)
    if pretrain_models:
        content.append('### Pretrained models\n')
        content.append(
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
        'Image Caption',
        'Visual Grounding',
        'Visual Question Answering',
        'Image-To-Text Retrieval',
        'Text-To-Image Retrieval',
        'NLVR',
    ]

    for task in tasks:
        task_models = filter_models_by_task(models, task=task)
        if task_models:
            datasets = {model.results[0].dataset for model in task_models}
            datasets = sorted(
                list(datasets), key=lambda x: DATASET_PRIORITY.get(x, 50))
            for dataset in datasets:
                content.append(f'### {task} on {dataset}\n')
                dataset_models = [
                    model for model in task_models
                    if model.results[0].dataset == dataset
                ]
                content.append(
                    generate_model_table(
                        dataset_models,
                        algo_folder,
                        pretrained_models=pretrain_models))
    return '\n'.join(content)


def parse_readme(readme):
    with open(readme, 'r') as f:
        file = f.read()

    content = {}

    for img_match in re.finditer(
            '^<div.*\n.*\n</div>\n', file, flags=re.MULTILINE):
        content['image'] = img_match.group()
        start, end = img_match.span()
        file = file[:start] + file[end:]
        break

    sections = re.split('^## ', file, flags=re.MULTILINE)
    for section in sections:
        if section.startswith('# '):
            content['title'] = section.strip() + '\n'
        elif section.startswith('Introduction'):
            content['intro'] = '## ' + section.strip() + '\n'
        elif section.startswith('Abstract'):
            content['abs'] = '## ' + section.strip() + '\n'
        elif section.startswith('How to use it'):
            content['usage'] = '## ' + section.strip() + '\n'
        elif section.startswith('Models and results'):
            content['models'] = '## ' + section.strip() + '\n'
        elif section.startswith('Citation'):
            content['citation'] = '## ' + section.strip() + '\n'
        else:
            section_title = section.split('\n', maxsplit=1)[0]
            content[section_title] = '## ' + section.strip() + '\n'
    return content


def combine_readme(content: dict):
    content = content.copy()
    readme = content.pop('title')
    if 'intro' in content:
        readme += f"\n{content.pop('intro')}"
        readme += f"\n{content.pop('image')}"
        readme += f"\n{content.pop('abs')}"
    else:
        readme += f"\n{content.pop('abs')}"
        readme += f"\n{content.pop('image')}"

    readme += f"\n{content.pop('usage')}"
    readme += f"\n{content.pop('models')}"

    citation = content.pop('citation')
    if content:
        # Custom sections
        for v in content.values():
            readme += f'\n{v}'
    readme += f'\n{citation}'
    return readme


def main():
    args = parse_args()
    metafile = load(str(args.metafile))
    if args.table:
        print(add_models(metafile))
        return

    if args.update is not None:
        content = parse_readme(args.update)
    else:
        content = {}

    if 'title' not in content or 'title' in args.update_items:
        content['title'] = add_title(metafile)
    if 'abs' not in content or 'abs' in args.update_items:
        content['abs'] = add_abstract(metafile)
    if 'image' not in content or 'image' in args.update_items:
        img = '<div align=center>\n<img src="" width="50%"/>\n</div>\n'
        content['image'] = img
    if 'usage' not in content or 'usage' in args.update_items:
        content['usage'] = add_usage(metafile)
    if 'models' not in content or 'models' in args.update_items:
        content['models'] = add_models(metafile)
    if 'citation' not in content:
        content['citation'] = '## Citation\n```bibtex\n```\n'

    content = combine_readme(content)
    if args.out is not None:
        with open(args.out, 'w') as f:
            f.write(content)
    else:
        print(content)


if __name__ == '__main__':
    main()
