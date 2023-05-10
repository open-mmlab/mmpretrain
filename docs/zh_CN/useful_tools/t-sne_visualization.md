# t-分布随机邻域嵌入（t-SNE）可视化

## t-分布随机邻域嵌入可视化工具介绍

MMPretrain 提供 `tools/visualization/vis_tsne.py` 工具来用t-SNE可视化图像的特征嵌入。请使用 `pip install scikit-learn` 安装 `sklearn` 来计算t-SNE。

**命令**：

```bash
python tools/visualization/vis_tsne.py \
    CONFIG \
    [--checkpoint CHECKPOINT] \
    [--work-dir WORK_DIR] \
    [--test-cfg TEST_CFG] \
    [--vis-stage {backbone,neck,pre_logits}]
    [--class-idx ${CLASS_IDX} [CLASS_IDX ...]]
    [--max-num-class MAX_NUM_CLASS]
    [--max-num-samples MAX_NUM_SAMPLES]
    [--cfg-options CFG_OPTIONS [CFG_OPTIONS ...]]
    [--device DEVICE]
    [--legend]
    [--show]
    [--n-components N_COMPONENTS]
    [--perplexity PERPLEXITY]
    [--early-exaggeration EARLY_EXAGGERATION]
    [--learning-rate LEARNING_RATE]
    [--n-iter N_ITER]
    [--n-iter-without-progress N_ITER_WITHOUT_PROGRESS]
    [--init INIT]
```

**所有参数的说明**：

- `CONFIG`: t-SNE 配置文件的路径。
- `--checkpoint CHECKPOINT`: 模型权重文件的路径。
- `--work-dir WORK_DIR`: 保存日志和可视化图像的目录。
- `--test-cfg TEST_CFG`: 用来加载 test_dataloader 配置的 t-SNE 配置文件的路径。
- `--vis-stage {backbone,neck,pre_logits}`: 模型可视化的阶段。
- `--class-idx CLASS_IDX [CLASS_IDX ...]`: 用来计算 t-SNE 的类别。
- `--max-num-class MAX_NUM_CLASS`: 前 N 个被应用 t-SNE 算法的类别，默认为20。
- `--max-num-samples MAX_NUM_SAMPLES`: 每个类别中最大的样本数，值越高需要的计算时间越长，默认为100。
- `--cfg-options CFG_OPTIONS [CFG_OPTIONS ...]`: 覆盖被使用的配置中的一些设定，xxx=yyy 格式的关键字-值对会被合并到配置文件中。override some settings in the used config, the key-value pair in xxx=yyy format will be merged into config file. If the value to be overwritten is a list, it should be like key="[a,b]" or key=a,b It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" Note that the quotation marks are necessary and that no white space is allowed.
- `--device DEVICE`: 用于推理的设备。
- `--legend`: 显示所有类别的图例。
- `--show`: 在图形窗口中显示结果。
- `--n-components N_COMPONENTS`: 结果的维数。
- `--perplexity PERPLEXITY`: 复杂度与其他流形学习算法中使用的最近邻的数量有关。
- `--early-exaggeration EARLY_EXAGGERATION`: 控制原空间中的自然聚类在嵌入空间中的紧密程度以及它们之间的空间大小。
- `--learning-rate LEARNING_RATE`: t-SNE 的学习率通常在[10.0, 1000.0]的范围内。如果学习率太高，数据可能看起来像一个球，其中任何一点与它最近的邻居近似等距。如果学习率太低，大多数点可能看起来被压缩在一个几乎没有异常值的密集点云中。
- `--n-iter N_ITER`: 优化的最大迭代次数。应该至少为250。
- `--n-iter-without-progress N_ITER_WITHOUT_PROGRESS`: 在我们中止优化之前，最大的没有进展的迭代次数。
- `--init INIT`: 初始化方法。

## 如何可视化 CNN 分类器的t-分布随机邻域嵌入（如 ResNet-18 和 ResNet-50）

以下是在CIFAR-10数据集上训练的 ResNet-18 和 ResNet-50 模型上运行t-SNE可视化的两个样例：

```shell
python tools/visualization/vis_tsne.py \
    configs/resnet/resnet18_8xb16_cifar10.py \
    --checkpoint https://download.openmmlab.com/mmclassification/v0/resnet/resnet18_b16x8_cifar10_20210528-bd6371c8.pth

python tools/visualization/vis_tsne.py \
    configs/resnet/resnet50_8xb16_cifar10.py \
    --checkpoint https://download.openmmlab.com/mmclassification/v0/resnet/resnet50_b16x8_cifar10_20210528-f54bfad9.pth
```

| ResNet-18                                                                                            | ResNet-50                                                                                            |
| ---------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------- |
| <div align=center><img src='https://user-images.githubusercontent.com/42371271/236410521-c4d087da-d16f-48ad-b951-c74d10c68f33.png' height="auto" width="auto" ></div> | <div align=center><img src='https://user-images.githubusercontent.com/42371271/236411844-c97dc514-dad0-401e-ba8f-307d0a385b4e.png' height="auto" width="auto" ></div> |

## 如何可视化自监督视觉 transformer 的t-分布随机邻域嵌入

待添加。
