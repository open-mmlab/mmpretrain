# 日志分析工具

## 日志分析

### 日志分析工具介绍

`tools/analysis_tools/analyze_logs.py` 脚本绘制指定键值的变化曲线。

<div align=center><img src="../_static/image/tools/analysis/analyze_log.jpg" style=" width: 75%; height: 30%; "></div>

```shell
python tools/analysis_tools/analyze_logs.py plot_curve  \
    ${JSON_LOGS}  \
    [--keys ${KEYS}]  \
    [--title ${TITLE}]  \
    [--legend ${LEGEND}]  \
    [--backend ${BACKEND}]  \
    [--style ${STYLE}]  \
    [--out ${OUT_FILE}]  \
    [--window-size ${WINDOW_SIZE}]
```

**所有参数的说明：**

- `json_logs` : 模型配置文件的路径（可同时传入多个，使用空格分开）。
- `--keys` : 分析日志的关键字段，数量为 `len(${JSON_LOGS}) * len(${KEYS})` 默认为 ‘loss’。
- `--title` : 分析日志的图片名称，默认使用配置文件名， 默认为空。
- `--legend` : 图例的名称，其数目必须与相等`len(${JSON_LOGS}) * len(${KEYS})`。     默认使用 `"${JSON_LOG}-${KEYS}"`.
- `--backend` : matplotlib 的绘图后端，默认由 matplotlib 自动选择。
- `--style` : 绘图配色风格，默认为 `whitegrid`。
- `--out` : 保存分析图片的路径，如不指定则不保存。
- `--window-size`: 可视化窗口大小，如果没有指定，默认为  `'12*7'`。如果需要指定，需按照格式 `'W*H'`。  

```{note}
The `--style` option depends on `seaborn` package, please install it before setting it.
```

### 如何或绘制损失/精度曲线

我们将给出一些示例，来展示如何使用 `tools/analysis_tools/analyze_logs.py`脚本绘制精度曲线的损失曲线

#### 绘制某日志文件对应的损失曲线图

```shell
python tools/analysis_tools/analyze_logs.py plot_curve your_log_json --keys loss --legend loss
```

#### 绘制某日志文件对应的 top-1 和 top-5 准确率曲线图，并将曲线图导出为 results.jpg 文件。

```shell
python tools/analysis_tools/analyze_logs.py plot_curve your_log_json --keys accuracy_top-1 accuracy_top-5  --legend top1 top5 --out results.jpg
```

#### 在同一图像内绘制两份日志文件对应的 top-1 准确率曲线图。

```shell
python tools/analysis_tools/analyze_logs.py plot_curve log1.json log2.json --keys accuracy_top-1 --legend exp1 exp2
```

```{note}
The tool will automatically select to find keys in training logs or validation logs according to the keys.
Therefore, if you add a custom evaluation metric, please also add the key to `TEST_METRICS` in this tool.
```

### 如何统计训练时间

`tools/analysis_tools/analyze_logs.py` 也可以根据日志文件统计训练耗时。

```shell
python tools/analysis_tools/analyze_logs.py cal_train_time \
    ${JSON_LOGS}
    [--include-outliers]
```

**所有参数的说明：**

- `json_logs` : 模型配置文件的路径（可同时传入多个，使用空格分开）。
- `--include-outliers` :如果指定，将不会排除每个轮次中第一轮迭代的记录（有时第一轮迭代会耗时较长）。

**示例：**

```shell
python tools/analysis_tools/analyze_logs.py cal_train_time work_dirs/some_exp/20200422_153324.log.json
```

预计输出结果如下所示：

```text
-----Analyze train time of work_dirs/some_exp/20200422_153324.log.json-----
slowest epoch 68, average time is 0.3818
fastest epoch 1, average time is 0.3694
time std over epochs is 0.0020
average iter time: 0.3777 s/iter
```

## 结果分析

利用  `tools/test.py`的`--out` ，w我们可以将所有的样本的推理结果保存到输出 文件中。利用这一文件，我们可以进行进一步的分析。

### 如何进行离线度量评估

我们提供了 `tools/analysis_tools/eval_metric.py` 脚本，使用户能够根据预测文件评估模型。

```shell
python tools/analysis_tools/eval_metric.py \
      ${CONFIG} \
      ${RESULT} \
      [--metrics ${METRICS}]  \
      [--cfg-options ${CFG_OPTIONS}] \
      [--metric-options ${METRIC_OPTIONS}]
```

**所有参数说明**：

- `config` : 配置文件的路径。
- `result`:  `tools/test.py`的输出结果文件。
- `--metrics` : 评估的衡量指标，可接受的值取决于数据集类。
- `--cfg-options`:额外的配置选项，会被合入配置文件，参考[教程 1：如何编写配置文件](https://mmclassification.readthedocs.io/zh_CN/latest/tutorials/config.html)。
- `--metric-options`:如果指定了，这些选项将被传递给数据集 `evaluate` 函数的 `metric_options` 参数。

```{note}
In `tools/test.py`, we support using `--out-items` option to select which kind of results will be saved. Please ensure the result file includes "class_scores" to use this tool.
```

**示例**:

```shell
python tools/analysis_tools/eval_metric.py configs/t2t_vit/t2t-vit-t-14_8xb64_in1k.py your_result.pkl --metrics accuracy --metric-options "topk=(1,5)"
```

### 如何将预测结果可视化

我们可以使用脚本 `tools/analysis_tools/analyze_results.py` 来保存预测成功或失败时得分最高的图像。

```shell
python tools/analysis_tools/analyze_results.py \
      ${CONFIG} \
      ${RESULT} \
      [--out-dir ${OUT_DIR}] \
      [--topk ${TOPK}] \
      [--cfg-options ${CFG_OPTIONS}]
```

**所有参数说明：**:

- `config` : 配置文件的路径。
- `result`:  `tools/test.py`的输出结果文件。
- `--out_dir`:保存结果分析的文件夹路径。
- `--topk`: 分别保存多少张预测成功/失败的图像。如果不指定，默认为 `20`。
- `--cfg-options`: 额外的配置选项，会被合入配置文件，参考[教程 1：如何编写配置文件](https://mmclassification.readthedocs.io/zh_CN/latest/tutorials/config.html)。

```{note}
In `tools/test.py`, we support using `--out-items` option to select which kind of results will be saved. Please ensure the result file includes "pred_score", "pred_label" and "pred_class" to use this tool.
```

**示例**:

```shell
python tools/analysis_tools/analyze_results.py \
       configs/resnet/resnet50_b32x8_imagenet.py \
       result.pkl \
       --out_dir results \
       --topk 50
```
