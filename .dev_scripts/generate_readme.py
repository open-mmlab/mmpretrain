# flake8: noqa
import argparse
import warnings
from collections import defaultdict
from pathlib import Path

from modelindex.load_model_index import load
from modelindex.models.ModelIndex import ModelIndex

prog_description = """\
Use metafile to generate a README.md.

Notice that the tool may fail in some corner cases, and you still need to check and fill some contents manually in the generated README.
"""


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
    readme.append(f'> [{title}]({url})')
    readme.append(f'<!-- [{papertype.upper()}] -->')
    readme.append('')


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


def add_models(metafile, readme):
    models = metafile.models
    if len(models) == 0:
        return

    readme.append('## Results and models')
    readme.append('')

    datasets = defaultdict(list)
    for model in models:
        if model.results is None:
            # No results on pretrained model.
            datasets['Pre-trained Models'].append(model)
        else:
            datasets[model.results[0].dataset].append(model)

    for dataset, models in datasets.items():
        if dataset == 'Pre-trained Models':
            readme.append(f'### {dataset}\n')
            readme.append(
                'The pre-trained models are only used to fine-tune, '
                "and therefore cannot be trained and don't have evaluation results.\n"
            )
            readme.append(
                '|         Model         |  Pretrain | Params(M) | Flops(G) | Config | Download |\n'
                '|:---------------------:|:---------:|:---------:|:--------:|:------:|:--------:|'
            )
            converted_from = None
            for model in models:
                name = model.name.center(21)
                params = model.metadata.parameters / 1e6
                flops = model.metadata.flops / 1e9
                converted_from = converted_from or model.data.get(
                    'Converted From', None)
                config = './' + Path(model.config).name
                weights = model.weights
                star = '\*' if '3rdparty' in weights else ''
                readme.append(
                    f'| {name}{star} | {params:.2f} | {flops:.2f} | [config]({config}) | [model]({weights}) |'
                ),
            if converted_from is not None:
                readme.append('')
                readme.append(
                    f"*Models with \* are converted from the [official repo]({converted_from['Code']}).*\n"
                )
        else:
            readme.append(f'### {dataset}\n')
            readme.append(
                '|         Model         |  Pretrain  | Params(M) | Flops(G) | Top-1 (%) | Top-5 (%) | Config | Download |\n'
                '|:---------------------:|:----------:|:---------:|:--------:|:---------:|:---------:|:------:|:--------:|'
            )
            converted_from = None
            for model in models:
                name = model.name.center(21)
                params = model.metadata.parameters / 1e6
                flops = model.metadata.flops / 1e9
                metrics = model.results[0].metrics
                top1 = metrics.get('Top 1 Accuracy')
                top5 = metrics.get('Top 5 Accuracy', 0)
                converted_from = converted_from or model.data.get(
                    'Converted From', None)
                config = './' + Path(model.config).name
                weights = model.weights
                star = '\*' if '3rdparty' in weights else ''
                if 'in21k-pre' in weights:
                    pretrain = 'ImageNet 21k'
                else:
                    pretrain = 'From scratch'
                readme.append(
                    f'| {name}{star} | {pretrain} | {params:.2f} | {flops:.2f} | {top1:.2f} | {top5:.2f} | [config]({config}) | [model]({weights}) |'
                ),
            if converted_from is not None:
                readme.append('')
                readme.append(
                    f"*Models with \* are converted from the [official repo]({converted_from['Code']}). "
                    'The config files of these models are only for inference. '
                    "We don't ensure these config files' training accuracy and welcome you to contribute your reproduction results.*\n"
                )


def main():
    args = parse_args()
    metafile = load(str(args.metafile))
    readme_lines = []
    if not args.table:
        add_title(metafile, readme_lines)
        add_abstract(metafile, readme_lines)
    add_models(metafile, readme_lines)
    if not args.table:
        readme_lines.append('## Citation\n')
        readme_lines.append('```bibtex\n\n```\n')
    print('\n'.join(readme_lines))


if __name__ == '__main__':
    main()
