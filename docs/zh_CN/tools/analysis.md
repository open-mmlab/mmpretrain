# 分析

<!-- TOC -->

- [日志分析](#日志分析)
  - [绘制曲线图](#绘制曲线图)
  - [统计训练时间](#统计训练时间)
- [结果分析](#结果分析)
  - [评估结果](#查看典型结果)
  - [查看典型结果](#查看典型结果)
- [模型复杂度分析](#模型复杂度分析)
- [常见问题](#常见问题)

<!-- TOC -->

## 日志分析

### 绘制曲线图

指定一个训练日志文件，可通过 `tools/analysis_tools/analyze_logs.py` 脚本绘制指定键值的变化曲线

<div align=center><img src="../_static/image/tools/analysis/analyze_log.jpg" style=" width: 75%; height: 30%; "></div>

```shell
python tools/analysis_tools/analyze_logs.py plot_curve \
    ${JSON_LOGS}  \
    [--keys ${KEYS}]  \
    [--title ${TITLE}]  \
    [--legend ${LEGEND}]  \
    [--backend ${BACKEND}]  \
    [--style ${STYLE}]  \
    [--out ${OUT_FILE}] \
    [--window-size ${WINDOW_SIZE}]
```

所有参数的说明

- `json_logs` ：模型配置文件的路径（可同时传入多个，使用空格分开）。
- `--keys` ：分析日志的关键字段，数量为 `len(${JSON_LOGS}) * len(${KEYS})` 默认为 'loss'。
- `--title` ：分析日志的图片名称，默认使用配置文件名， 默认为空。
- `--legend` ：图例名（可同时传入多个，使用空格分开，数目与 `${JSON_LOGS} * ${KEYS}` 数目一致）。默认使用 `"${JSON_LOG}-${KEYS}"`。
- `--backend` ：matplotlib 的绘图后端，默认由 matplotlib 自动选择。
- `--style` ：绘图配色风格，默认为 `whitegrid`。
- `--out` ：保存分析图片的路径，如不指定则不保存。
- `--window-size`: 可视化窗口大小，如果没有指定，默认为 `12*7`。如果需要指定，需按照格式 `'W*H'`。

```{note}
`--style` 选项依赖于第三方库 `seaborn`，需要设置绘图风格请现安装该库。
```

例如:

- 绘制某日志文件对应的损失曲线图。

    ```shell
    python tools/analysis_tools/analyze_logs.py plot_curve your_log_json --keys loss --legend loss
    ```

- 绘制某日志文件对应的 top-1 和 top-5 准确率曲线图，并将曲线图导出为 results.jpg 文件。

    ```shell
    python tools/analysis_tools/analyze_logs.py plot_curve your_log_json --keys accuracy_top-1 accuracy_top-5  --legend top1 top5 --out results.jpg
    ```

- 在同一图像内绘制两份日志文件对应的 top-1 准确率曲线图。

    ```shell
    python tools/analysis_tools/analyze_logs.py plot_curve log1.json log2.json --keys accuracy_top-1 --legend run1 run2
    ```

```{note}
本工具会自动根据关键字段选择从日志的训练部分还是验证部分读取，因此如果你添加了
自定义的验证指标，请把相对应的关键字段加入到本工具的 `TEST_METRICS` 变量中。
```

### 统计训练时间

`tools/analysis_tools/analyze_logs.py` 也可以根据日志文件统计训练耗时。

```shell
python tools/analysis_tools/analyze_logs.py cal_train_time \
    ${JSON_LOGS}
    [--include-outliers]
```

**所有参数的说明**：

- `json_logs` ：模型配置文件的路径（可同时传入多个，使用空格分开）。
- `--include-outliers` ：如果指定，将不会排除每个轮次中第一轮迭代的记录（有时第一轮迭代会耗时较长）

**示例**:

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

利用 `tools/test.py` 的 `--out` 参数，我们可以将所有的样本的推理结果保存到输出
文件中。利用这一文件，我们可以进行进一步的分析。

### 评估结果

`tools/analysis_tools/eval_metric.py` 可以用来再次计算评估结果。

```shell
python tools/analysis_tools/eval_metric.py \
      ${CONFIG} \
      ${RESULT} \
      [--metrics ${METRICS}]  \
      [--cfg-options ${CFG_OPTIONS}] \
      [--metric-options ${METRIC_OPTIONS}]
```

**所有参数说明**：

- `config` ：配置文件的路径。
- `result` ： `tools/test.py` 的输出结果文件。
- `metrics` ： 评估的衡量指标，可接受的值取决于数据集类。
- `--cfg-options`: 额外的配置选项，会被合入配置文件，参考[教程 1：如何编写配置文件](https://mmclassification.readthedocs.io/zh_CN/latest/tutorials/config.html)。
- `--metric-options`: 如果指定了，这些选项将被传递给数据集 `evaluate` 函数的 `metric_options` 参数。

```{note}
在 `tools/test.py` 中，我们支持使用 `--out-items` 选项来选择保存哪些结果。为了使用本工具，请确保结果文件中包含 "class_scores"。
```

**示例**：

```shell
python tools/analysis_tools/eval_metric.py configs/t2t_vit/t2t-vit-t-14_8xb64_in1k.py ./result.pkl --metrics accuracy --metric-options "topk=(1,5)"
```

### 查看典型结果

`tools/analysis_tools/analyze_results.py` 可以保存预测成功/失败，同时得分最高的 k 个图像。

```shell
python tools/analysis_tools/analyze_results.py \
      ${CONFIG} \
      ${RESULT} \
      [--out-dir ${OUT_DIR}] \
      [--topk ${TOPK}] \
      [--cfg-options ${CFG_OPTIONS}]
```

**所有参数说明**：

- `config` ：配置文件的路径。
- `result` ： `tools/test.py` 的输出结果文件。
- `--out_dir` ：保存结果分析的文件夹路径。
- `--topk` ：分别保存多少张预测成功/失败的图像。如果不指定，默认为 `20`。
- `--cfg-options`: 额外的配置选项，会被合入配置文件，参考[教程 1：如何编写配置文件](https://mmclassification.readthedocs.io/zh_CN/latest/tutorials/config.html)。

```{note}
在 `tools/test.py` 中，我们支持使用 `--out-items` 选项来选择保存哪些结果。为了使用本工具，请确保结果文件中包含 "pred_score"、"pred_label" 和 "pred_class"。
```

**示例**：

```shell
python tools/analysis_tools/analyze_results.py \
       configs/resnet/resnet50_xxxx.py \
       result.pkl \
       --out_dir results \
       --topk 50
```

## 模型复杂度分析

### 计算 FLOPs 和参数量（试验性的）

我们根据 [flops-counter.pytorch](https://github.com/sovrasov/flops-counter.pytorch) 提供了一个脚本用于计算给定模型的 FLOPs 和参数量。

```shell
python tools/analysis_tools/get_flops.py ${CONFIG_FILE} [--shape ${INPUT_SHAPE}]
```

**所有参数说明**：

- `config` ：配置文件的路径。
- `--shape`: 输入尺寸，支持单值或者双值， 如： `--shape 256`、`--shape 224 256`。默认为`224 224`。

用户将获得如下结果：

```text
==============================
Input shape: (3, 224, 224)
Flops: 4.12 GFLOPs
Params: 25.56 M
==============================
```

```{warning}
此工具仍处于试验阶段，我们不保证该数字正确无误。您最好将结果用于简单比较，但在技术报告或论文中采用该结果之前，请仔细检查。
- FLOPs 与输入的尺寸有关，而参数量与输入尺寸无关。默认输入尺寸为 (1, 3, 224, 224)
- 一些运算不会被计入 FLOPs 的统计中，例如 GN 和自定义运算。详细信息请参考 [`mmcv.cnn.get_model_complexity_info()`](https://github.com/open-mmlab/mmcv/blob/master/mmcv/cnn/utils/flops_counter.py)
```

## 常见问题

- 无
