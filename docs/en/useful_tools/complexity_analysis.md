# Model Complexity Analysis

## Get the FLOPs and params (experimental)

We provide a script adapted from [MMEngine](https://github.com/open-mmlab/mmengine/blob/main/mmengine/analysis/complexity_analysis.py) to compute the FLOPs and params of a given model.

```shell
python tools/analysis_tools/get_flops.py ${CONFIG_FILE} [--shape ${INPUT_SHAPE}]
```

Description of all arguments:

- `config`: The path of the model config file.
- `--shape`: Input size, support single value or double value parameter, such as `--shape 256` or `--shape 224 256`. If not set, default to be `224 224`.

Example:

```shell
python tools/analysis_tools/get_flops.py configs/resnet/resnet50_8xb32_in1k.py
```

You will get the final result like this.

```text
==============================
Input shape: (3, 224, 224)
Flops: 4.109G
Params: 25.557M
Activation: 11.114M
==============================
```

Also, you will get the detailed complexity information of each layer like this:

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
This tool is still experimental and we do not guarantee that the number is correct. You may well use the result for simple comparisons, but double-check it before you adopt it in technical reports or papers.
- FLOPs are related to the input shape while parameters are not. The default input shape is (1, 3, 224, 224).
- Some operators are not counted into FLOPs like custom operators. Refer to [`mmengine.analysis.complexity_analysis._DEFAULT_SUPPORTED_FLOP_OPS`](https://github.com/open-mmlab/mmengine/blob/main/mmengine/analysis/complexity_analysis.py) for details.
```
