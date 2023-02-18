# 模型复杂度分析

## 计算FLOPs 和参数数量（实验性的）

我们根据 [fvcore](https://github.com/facebookresearch/fvcore/blob/main/fvcore/nn/flop_count.py) 提供了一个脚本用于计算给定模型的 FLOPs 和参数量。

```shell
python tools/analysis_tools/get_flops.py ${CONFIG_FILE} [--shape ${INPUT_SHAPE}]
```

所有参数说明：

- `config` : 配置文件的路径。
- `--shape`: 输入尺寸，支持单值或者双值， 如： `--shape 256`、`--shape 224 256`。默认为`224 224`。

你将获得如下结果：

```text
==============================
Input shape: (3, 224, 224)
Flops: 17.582G
Params: 91.234M
Activation: 23.895M
==============================
```

同时，你会得到每层的详细复杂度信息，如下所示:

```text
| module                                    | #parameters or shape   | #flops    | #activations   |
|:------------------------------------------|:-----------------------|:----------|:---------------|
| model                                     | 91.234M                | 17.582G   | 23.895M        |
|  backbone                                 |  85.799M               |  17.582G  |  23.895M       |
|   backbone.cls_token                      |   (1, 1, 768)          |           |                |
|   backbone.pos_embed                      |   (1, 197, 768)        |           |                |
|   backbone.patch_embed.projection         |   0.591M               |   0.116G  |   0.151M       |
|    backbone.patch_embed.projection.weight |    (768, 3, 16, 16)    |           |                |
|    backbone.patch_embed.projection.bias   |    (768,)              |           |                |
|   backbone.layers                         |   85.054M              |   17.466G |   23.744M      |
|    backbone.layers.0                      |    7.088M              |    1.455G |    1.979M      |
|    backbone.layers.1                      |    7.088M              |    1.455G |    1.979M      |
|    backbone.layers.2                      |    7.088M              |    1.455G |    1.979M      |
|    backbone.layers.3                      |    7.088M              |    1.455G |    1.979M      |
|    backbone.layers.4                      |    7.088M              |    1.455G |    1.979M      |
|    backbone.layers.5                      |    7.088M              |    1.455G |    1.979M      |
|    backbone.layers.6                      |    7.088M              |    1.455G |    1.979M      |
|    backbone.layers.7                      |    7.088M              |    1.455G |    1.979M      |
|    backbone.layers.8                      |    7.088M              |    1.455G |    1.979M      |
|    backbone.layers.9                      |    7.088M              |    1.455G |    1.979M      |
|    backbone.layers.10                     |    7.088M              |    1.455G |    1.979M      |
|    backbone.layers.11                     |    7.088M              |    1.455G |    1.979M      |
|   backbone.ln1                            |   1.536K               |   0.756M  |   0            |
|    backbone.ln1.weight                    |    (768,)              |           |                |
|    backbone.ln1.bias                      |    (768,)              |           |                |
|  head.layers                              |  5.435M                |           |                |
|   head.layers.pre_logits                  |   2.362M               |           |                |
|    head.layers.pre_logits.weight          |    (3072, 768)         |           |                |
|    head.layers.pre_logits.bias            |    (3072,)             |           |                |
|   head.layers.head                        |   3.073M               |           |                |
|    head.layers.head.weight                |    (1000, 3072)        |           |                |
|    head.layers.head.bias                  |    (1000,)             |           |                |
```

```{warning}
警告

此工具仍处于试验阶段，我们不保证该数字正确无误。您最好将结果用于简单比较，但在技术报告或论文中采用该结果之前，请仔细检查。

- FLOPs 与输入的尺寸有关，而参数量与输入尺寸无关。默认输入尺寸为 (1, 3, 224, 224)
- 一些运算不会被计入 FLOPs 的统计中，例如某些自定义运算。详细信息请参考 [`fvcore.nn.flop_count._DEFAULT_SUPPORTED_OPS`](https://github.com/facebookresearch/fvcore/blob/main/fvcore/nn/flop_count.py)。
```
