## 形状偏差(Shape Bias)工具用法

形状偏差(shape bias)衡量模型与纹理相比，如何依赖形状来感知图像中的语义。关于更多细节，我们向感兴趣的读者推荐这篇[论文](https://arxiv.org/abs/2106.07411) 。MMPretrain提供现成的工具箱来获得分类模型的形状偏差。您可以按照以下步骤操作：

### 准备数据集

首先你应该下载[cue-conflict](https://github.com/bethgelab/model-vs-human/releases/download/v0.1/cue-conflict.tar.gz) 到`data`文件夹，然后解压缩这个数据集。之后，你的`data`文件夹应具有一下结构：

```text
data
├──cue-conflict
|      |──airplane
|      |──bear
|      ...
|      |── truck
```

### 修改分类配置

我们在使用MAE预训练的ViT-base模型上运行形状偏移工具。它的配置文件为`configs/mae/benchmarks/vit-base-p16_8xb128-coslr-100e_in1k.py`，它的检查点可从[此链接](https://download.openmmlab.com/mmselfsup/1.x/mae/mae_vit-base-p16_8xb512-fp16-coslr-1600e_in1k/vit-base-p16_ft-8xb128-coslr-100e_in1k/vit-base-p16_ft-8xb128-coslr-100e_in1k_20220825-cf70aa21.pth) 下载。将原始配置中的test_pipeline, test_dataloader和test_evaluation替换为以下配置：

```python
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='ResizeEdge',
        scale=256,
        edge='short',
        backend='pillow'),
    dict(type='CenterCrop', crop_size=224),
    dict(type='PackInputs')
]
test_dataloader = dict(
    pin_memory=True,
    collate_fn=dict(type='default_collate'),
    batch_size=32,
    num_workers=4,
    dataset=dict(
        type='CustomDataset',
        data_root='data/cue-conflict',
        pipeline=test_pipeline,
        _delete_=True),
    sampler=dict(type='DefaultSampler', shuffle=False),
    persistent_workers=True,
    drop_last=False)
test_evaluator = dict(
    type='mmpretrain.ShapeBiasMetric',
    _delete_=True,
    csv_dir='work_dirs/shape_bias',
    model_name='mae')
```

请注意，你应该对上面的`csv_dir`和`model_name`进行自定义修改。我把修改后的示例配置文件重命名为`configs/mae/benchmarks/`文件夹中的`vit-base-p16_8xb128-coslr-100e_in1k_shape-bias.py`文件。

### 用上面修改后的配置文件在你的模型上做推断

然后，你应该使用修改后的配置文件在`cue-conflict`数据集上推断你的模型。

```shell
# For PyTorch
bash tools/dist_test.sh $CONFIG $CHECKPOINT
```

**所有参数的说明**：

- `$CONFIG`: 修改后的配置文件的路径。
- `$CHECKPOINT`: 检查点文件的路径或链接。

```shell
# Example
bash tools/dist_test.sh configs/mae/benchmarks/vit-base-p16_8xb128-coslr-100e_in1k_shape-bias.py https://download.openmmlab.com/mmselfsup/1.x/mae/mae_vit-base-p16_8xb512-fp16-coslr-1600e_in1k/vit-base-p16_ft-8xb128-coslr-100e_in1k/vit-base-p16_ft-8xb128-coslr-100e_in1k_20220825-cf70aa21.pth 1
```

之后，你应该在`csv_dir`文件夹中获得一个名为`cue-conflict_model-name_session-1.csv`的csv文件。除了这个文件以外，你还应该下载这些[csv文件](https://github.com/bethgelab/model-vs-human/tree/master/raw-data/cue-conflict) 到`csv_dir`。

### 绘制形状偏差图

然后我们可以开始绘制形状偏差图：

```shell
python tools/analysis_tools/shape_bias.py --csv-dir $CSV_DIR --result-dir $RESULT_DIR --colors $RGB --markers o --plotting-names $YOUR_MODEL_NAME --model-names $YOUR_MODEL_NAME
```

**所有参数的说明**:

- `--csv-dir $CSV_DIR`, 与保存这些csv文件的目录相同。
- `--result-dir $RESULT_DIR`, 输出名为`cue-conflict_shape-bias_matrixplot.pdf`的结果的目录。
- `--colors $RGB`, 应该是RGB值，格式为R G B，例如100 100 100，如果你想绘制几个模型的形状偏差，可以是多个RGB值。
- `--plotting-names $YOUR_MODEL_NAME`, 形状偏移图中图例的名称，您可以将其设置为模型名称。如果要绘制多个模型，plotting_names可以是多个值。
- `model-names $YOUR_MODEL_NAME`, 应与配置中指定的名称相同，如果要绘制多个模型的形状偏差，则可以是多个名称。

请注意，`--colors`的每三个值对应于`--model-names`的一个值。完成以上所有步骤后，你将获得下图。

<div align="center">
<img src="https://github.com/open-mmlab/mmpretrain/assets/42371271/dc608d06-43eb-4860-bb70-486ed2a3f927" width="500" />
</div>
