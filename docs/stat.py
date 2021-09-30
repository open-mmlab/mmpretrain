#!/usr/bin/env python
import functools as func
import glob
import os
import re
from pathlib import Path

import numpy as np

papers_root = Path('papers')
papers_root.mkdir(exist_ok=True)
files = [Path(f) for f in sorted(glob.glob('../configs/*/README.md'))]

stats = []
titles = []
num_ckpts = 0
num_configs = 0

for f in files:
    url = papers_root / (f.parent.name + '.md')
    if url.exists():
        os.remove(url)

    with open(f, 'r') as content_file:
        content = content_file.read()

    title = content.split('\n')[0].replace('# ', '').strip()
    abbr = [x for x in re.findall(r'<!-- {(.+)} -->', content)]
    abbr = abbr[0] if len(abbr) > 0 else title

    ckpts = set(x.lower().strip()
                for x in re.findall(r'\[model\]\((https?.*)\)', content))

    if len(ckpts) == 0:
        continue

    url.symlink_to(f.absolute())

    _papertype = [x for x in re.findall(r'\[([A-Z]+)\]', content)]
    assert len(_papertype) > 0
    papertype = _papertype[0]

    paper = set([(papertype, title)])

    num_ckpts += len(ckpts)
    titles.append(title)

    statsmsg = f"""
\t* [{papertype}] [{title}]({url}) ({len(ckpts)} ckpts)
"""
    stats.append(
        dict(paper=paper, ckpts=ckpts, statsmsg=statsmsg, abbr=abbr, url=url))

allpapers = func.reduce(lambda a, b: a.union(b), [stat["paper"] for stat in stats])
msglist = '\n'.join(stat["statsmsg"] for stat in stats)

papertypes, papercounts = np.unique([t for t, _ in allpapers],
                                    return_counts=True)
countstr = '\n'.join(
    [f'   - {t}: {c}' for t, c in zip(papertypes, papercounts)])

modelzoo = f"""
# Model Zoo Summary

* Number of papers: {len(set(titles))}
{countstr}

* Number of checkpoints: {num_ckpts}
{msglist}
"""

with open('modelzoo_statistics.md', 'w') as f:
    f.write(modelzoo)

toctree = """
.. toctree::
   :maxdepth: 1
   :caption: Model zoo
   :glob:

   modelzoo_statistics.md
   model_zoo.md
"""
with open('_model_zoo.rst', 'w') as f:
    f.write(toctree)
    for stat in stats:
        f.write(f'   {stat["abbr"]} <{stat["url"]}>\n')
