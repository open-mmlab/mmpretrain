#!/usr/bin/env python
import re
import warnings
from collections import defaultdict
from pathlib import Path

from modelindex.load_model_index import load
from modelindex.models.Result import Result
from tabulate import tabulate

MMPT_ROOT = Path(__file__).absolute().parents[2]
PAPERS_ROOT = Path('papers')  # Path to save generated paper pages.
GITHUB_PREFIX = 'https://github.com/open-mmlab/mmpretrain/blob/main/'
MODELZOO_TEMPLATE = """\
# Model Zoo Summary

In this page, we list [all algorithms](#all-supported-algorithms) we support. You can click the link to jump to the corresponding model pages.

And we also list all checkpoints for different tasks we provide. You can sort or search checkpoints in the table and click the corresponding link to model pages for more details.

## All supported algorithms

* Number of papers: {num_papers}
{type_msg}

* Number of checkpoints: {num_ckpts}
{paper_msg}

"""  # noqa: E501

METRIC_ALIAS = {
    'Top 1 Accuracy': 'Top-1 (%)',
    'Top 5 Accuracy': 'Top-5 (%)',
}

model_index = load(str(MMPT_ROOT / 'model-index.yml'))


def build_collections(model_index):
    col_by_name = {}
    for col in model_index.collections:
        setattr(col, 'models', [])
        col_by_name[col.name] = col

    for model in model_index.models:
        col = col_by_name[model.in_collection]
        col.models.append(model)
        setattr(model, 'collection', col)
        if model.results is None:
            setattr(model, 'tasks', [])
        else:
            setattr(model, 'tasks', [result.task for result in model.results])


build_collections(model_index)


def count_papers(collections):
    total_num_ckpts = 0
    type_count = defaultdict(int)
    paper_msgs = []

    for collection in collections:
        with open(MMPT_ROOT / collection.readme) as f:
            readme = f.read()
        ckpts = set(x.lower().strip()
                    for x in re.findall(r'\[model\]\((https?.*)\)', readme))
        total_num_ckpts += len(ckpts)
        title = collection.paper['Title']
        papertype = collection.data.get('type', 'Algorithm')
        type_count[papertype] += 1

        readme = PAPERS_ROOT / Path(
            collection.filepath).parent.with_suffix('.md').name
        paper_msgs.append(
            f'\t- [{papertype}] [{title}]({readme}) ({len(ckpts)} ckpts)')

    type_msg = '\n'.join(
        [f'\t- {type_}: {count}' for type_, count in type_count.items()])
    paper_msg = '\n'.join(paper_msgs)

    modelzoo = MODELZOO_TEMPLATE.format(
        num_papers=len(collections),
        num_ckpts=total_num_ckpts,
        type_msg=type_msg,
        paper_msg=paper_msg,
    )

    with open('modelzoo_statistics.md', 'w') as f:
        f.write(modelzoo)


count_papers(model_index.collections)


def generate_paper_page(collection):
    PAPERS_ROOT.mkdir(exist_ok=True)

    # Write a copy of README
    with open(MMPT_ROOT / collection.readme) as f:
        readme = f.read()
    folder = Path(collection.filepath).parent
    copy = PAPERS_ROOT / folder.with_suffix('.md').name

    def replace_link(matchobj):
        # Replace relative link to GitHub link.
        name = matchobj.group(1)
        link = matchobj.group(2)
        if not link.startswith('http'):
            assert (folder / link).exists(), \
                f'Link not found:\n{collection.readme}: {link}'
            rel_link = (folder / link).absolute().relative_to(MMPT_ROOT)
            link = GITHUB_PREFIX + str(rel_link)
        return f'[{name}]({link})'

    content = re.sub(r'\[([^\]]+)\]\(([^)]+)\)', replace_link, readme)
    content = f'---\ngithub_page: /{collection.readme}\n---\n' + content

    def make_tabs(matchobj):
        """modify the format from emphasis black symbol to tabs."""
        content = matchobj.group()
        content = content.replace('<!-- [TABS-BEGIN] -->', '')
        content = content.replace('<!-- [TABS-END] -->', '')

        # split the content by "**{Tab-Name}**""
        splits = re.split(r'^\*\*(.*)\*\*$', content, flags=re.M)[1:]
        tabs_list = []
        for title, tab_content in zip(splits[::2], splits[1::2]):
            title = ':::{tab} ' + title + '\n'
            tab_content = tab_content.strip() + '\n:::\n'
            tabs_list.append(title + tab_content)

        return '::::{tabs}\n' + ''.join(tabs_list) + '::::'

    if '<!-- [TABS-BEGIN] -->' in content and '<!-- [TABS-END] -->' in content:
        # Make TABS block a selctive tabs
        try:
            pattern = r'<!-- \[TABS-BEGIN\] -->([\d\D]*?)<!-- \[TABS-END\] -->'
            content = re.sub(pattern, make_tabs, content)
        except Exception as e:
            warnings.warn(f'Can not parse the TABS, get an error : {e}')

    with open(copy, 'w') as copy_file:
        copy_file.write(content)


for collection in model_index.collections:
    generate_paper_page(collection)


def scatter_results(models):
    model_result_pairs = []
    for model in models:
        if model.results is None:
            result = Result(task=None, dataset=None, metrics={})
            model_result_pairs.append((model, result))
        else:
            for result in model.results:
                model_result_pairs.append((model, result))
    return model_result_pairs


def generate_summary_table(task, model_result_pairs, title=None):
    metrics = set()
    for model, result in model_result_pairs:
        if result.task == task:
            metrics = metrics.union(result.metrics.keys())
    metrics = sorted(list(metrics))

    rows = []
    for model, result in model_result_pairs:
        if result.task != task:
            continue
        name = model.name
        params = f'{model.metadata.parameters / 1e6:.2f}'  # Params
        flops = f'{model.metadata.flops / 1e9:.2f}'  # Params
        readme = Path(model.collection.filepath).parent.with_suffix('.md').name
        page = f'[link]({PAPERS_ROOT / readme})'
        model_metrics = []
        for metric in metrics:
            model_metrics.append(str(result.metrics.get(metric, '')))

        rows.append([name, params, flops, *model_metrics, page])

    with open('modelzoo_statistics.md', 'a') as f:
        if title is not None:
            f.write(f'\n{title}')
        f.write("""\n```{table}\n:class: model-summary\n""")
        header = [
            'Model',
            'Params (M)',
            'Flops (G)',
            *[METRIC_ALIAS.get(metric, metric) for metric in metrics],
            'Readme',
        ]
        table_cfg = dict(
            tablefmt='pipe',
            floatfmt='.2f',
            numalign='right',
            stralign='center')
        f.write(tabulate(rows, header, **table_cfg))
        f.write('\n```\n')


def generate_dataset_wise_table(task, model_result_pairs, title=None):
    dataset_rows = defaultdict(list)
    for model, result in model_result_pairs:
        if result.task == task:
            dataset_rows[result.dataset].append((model, result))

    if title is not None:
        with open('modelzoo_statistics.md', 'a') as f:
            f.write(f'\n{title}')
    for dataset, pairs in dataset_rows.items():
        generate_summary_table(task, pairs, title=f'### {dataset}')


model_result_pairs = scatter_results(model_index.models)

# Generate Pretrain Summary
generate_summary_table(
    task=None,
    model_result_pairs=model_result_pairs,
    title='## Pretrained Models',
)

# Generate Image Classification Summary
generate_dataset_wise_table(
    task='Image Classification',
    model_result_pairs=model_result_pairs,
    title='## Image Classification',
)

# Generate Multi-Label Classification Summary
generate_dataset_wise_table(
    task='Multi-Label Classification',
    model_result_pairs=model_result_pairs,
    title='## Multi-Label Classification',
)

# Generate Image Retrieval Summary
generate_dataset_wise_table(
    task='Image Retrieval',
    model_result_pairs=model_result_pairs,
    title='## Image Retrieval',
)
