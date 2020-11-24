#!/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
from adjustText import adjust_text
from collections import namedtuple

import matplotlib as mpl


def plot(
    xs,
    ys,
    ss,
    s_scale,
    ann,
    label,
    xlabel,
    dsname,
    xlim,
    ylim,
    fontsize,
    append=False,
    finish=True,
    legendloc='lower right',
):
    xs = np.array(xs, dtype=np.float)
    ys = np.array(ys, dtype=np.float)
    ss = np.array(ss, dtype=np.float) * s_scale

    if not append:
        fig = plt.figure()
        # plt.grid(True)

    plt.scatter(x=xs, y=ys, s=ss, label=label, edgecolors=(0, 0, 0, .1), alpha=0.7)

    texts = []
    for n, x, y in zip(ann, xs, ys):
        texts.append(plt.annotate(n, (x, y), size=fontsize))

    adjust_text(
        texts,
        x=xs,
        y=ys,
        # autoalign='y',
        # force_text=(0.1, 1),
        # force_points=(0.2, 2),
        # force_objects=(0.1, 1),
        arrowprops=dict(arrowstyle="->", color="r", lw=1),
    )

    if finish:
        # plt.legend(bbox_to_anchor=(0, 1), loc='upper left')
        plt.legend(loc=legendloc)
        plt.xlim(xlim)
        plt.ylim(ylim)
        # plt.xscale('log')
        plt.title(f"{label} accuracy on {dsname} / {xlabel}")
        plt.xlabel(xlabel)
        plt.ylabel("Accuracy")
        plt.savefig(f"{dsname}-{xlabel}-{label}.svg")


def main():
    ## parse data from md file
    with open("model_zoo.md") as f:
        lines = f.read().splitlines()

    graph_list = {}

    for line in lines:
        if line.startswith("##"):
            dsname = line[2:].strip()
            fig_data = []
            graph_list[dsname] = fig_data
        elif "[model]" in line:
            data = [s.strip() for s in line.split("|")][1:6]
            fig_data.append(data)

    # ## generate SVG figures
    modname, param, flops, e1, e5 = zip(*graph_list["CIFAR10"])
    plot(flops, e1, param, 5, modname, "top-1", "GFLOPS", "CIFAR10", [0.0, 4], [94, 97], 10)
    plot(param, e1, flops, 50, modname, "top-1", "MParams", "CIFAR10", [0.0, 70], [94, 97], 10)

    modname, param, flops, e1, e5 = zip(*graph_list["ImageNet"])
    plot(flops, e1, param, 5, modname, "top-1", "GFLOPS", "ImageNet", [0.0, 25], [65, 85], 8)
    plot(flops, e5, param, 5, modname, "top-5", "GFLOPS", "ImageNet", [0.0, 25], [86, 98], 8)
    plot(param, e1, flops, 30, modname, "top-1", "MParams", "ImageNet", [0.0, 160], [65, 85], 8)
    plot(param, e5, flops, 30, modname, "top-5", "MParams", "ImageNet", [0.0, 160], [86, 98], 8)


if __name__ == "__main__":
    mpl.style.use('seaborn-muted')
    main()
