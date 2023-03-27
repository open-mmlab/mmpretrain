# 模型复杂度分析

## 计算 FLOPs 和参数数量（实验性的）

我们根据 [MMEngine](https://github.com/open-mmlab/mmengine/blob/main/mmengine/analysis/complexity_analysis.py) 提供了一个脚本用于计算给定模型的 FLOPs 和参数量。

```shell
python tools/analysis_tools/get_flops.py ${CONFIG_FILE} [--shape ${INPUT_SHAPE}]
```

所有参数说明：

- `config` : 配置文件的路径。
- `--shape`: 输入尺寸，支持单值或者双值， 如： `--shape 256`、`--shape 224 256`。默认为`224 224`。

示例：

```shell
python tools/analysis_tools/get_flops.py configs/resnet/resnet50_8xb32_in1k.py
```

你将获得如下结果：

```text
==============================
Input shape: (3, 224, 224)
Flops: 4.109G
Params: 25.557M
Activation: 11.114M
==============================
```

同时，你会得到每层的详细复杂度信息，如下所示：

```text
+--------------------------+----------------------+-----------+--------------+
| module                   | #parameters or shape | #flops    | #activations |
+--------------------------+----------------------+-----------+--------------+
| model                    | 25.557M              | 4.109G    | 11.114M      |
|  backbone                |  23.508M             |  4.109G   |  11.114M     |
|   backbone.conv1         |   9.408K             |   0.118G  |   0.803M     |
|    backbone.conv1.weight |    (64, 3, 7, 7)     |           |              |
|   backbone.bn1           |   0.128K             |   1.606M  |   0          |
|    backbone.bn1.weight   |    (64,)             |           |              |
|    backbone.bn1.bias     |    (64,)             |           |              |
|   backbone.layer1        |   0.216M             |   0.677G  |   4.415M     |
|    backbone.layer1.0     |    75.008K           |    0.235G |    2.007M    |
|    backbone.layer1.1     |    70.4K             |    0.221G |    1.204M    |
|    backbone.layer1.2     |    70.4K             |    0.221G |    1.204M    |
|   backbone.layer2        |   1.22M              |   1.034G  |   3.111M     |
|    backbone.layer2.0     |    0.379M            |    0.375G |    1.305M    |
|    backbone.layer2.1     |    0.28M             |    0.22G  |    0.602M    |
|    backbone.layer2.2     |    0.28M             |    0.22G  |    0.602M    |
|    backbone.layer2.3     |    0.28M             |    0.22G  |    0.602M    |
|   backbone.layer3        |   7.098M             |   1.469G  |   2.158M     |
|    backbone.layer3.0     |    1.512M            |    0.374G |    0.652M    |
|    backbone.layer3.1     |    1.117M            |    0.219G |    0.301M    |
|    backbone.layer3.2     |    1.117M            |    0.219G |    0.301M    |
|    backbone.layer3.3     |    1.117M            |    0.219G |    0.301M    |
|    backbone.layer3.4     |    1.117M            |    0.219G |    0.301M    |
|    backbone.layer3.5     |    1.117M            |    0.219G |    0.301M    |
|   backbone.layer4        |   14.965M            |   0.81G   |   0.627M     |
|    backbone.layer4.0     |    6.04M             |    0.373G |    0.326M    |
|    backbone.layer4.1     |    4.463M            |    0.219G |    0.151M    |
|    backbone.layer4.2     |    4.463M            |    0.219G |    0.151M    |
|  head.fc                 |  2.049M              |           |              |
|   head.fc.weight         |   (1000, 2048)       |           |              |
|   head.fc.bias           |   (1000,)            |           |              |
|  neck.gap                |                      |  0.1M     |  0           |
+--------------------------+----------------------+-----------+--------------+
```

```{warning}
警告

此工具仍处于试验阶段，我们不保证该数字正确无误。您最好将结果用于简单比较，但在技术报告或论文中采用该结果之前，请仔细检查。

- FLOPs 与输入的尺寸有关，而参数量与输入尺寸无关。默认输入尺寸为 (1, 3, 224, 224)
- 一些运算不会被计入 FLOPs 的统计中，例如某些自定义运算。详细信息请参考 [`mmengine.analysis.complexity_analysis._DEFAULT_SUPPORTED_FLOP_OPS`](https://github.com/open-mmlab/mmengine/blob/main/mmengine/analysis/complexity_analysis.py)。
```
