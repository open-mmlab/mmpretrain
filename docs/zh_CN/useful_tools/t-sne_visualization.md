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
- `--cfg-options CFG_OPTIONS [CFG_OPTIONS ...]`: 覆盖被使用的配置中的一些设定，形如 xxx=yyy 格式的关键字-值对会被合并到配置文件中。如果被覆盖的值是一个列表，它应该形如 key="[a,b]" 或者 key=a,b 。它还允许嵌套的列表/元组值，例如 key="[(a,b),(c,d)]" 。注意引号是必需的，而且不允许有空格。
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

## 如何可视化分类模型的t-SNE（如 ResNet）

以下是在CIFAR-10数据集上训练的 ResNet-18 和 ResNet-50 模型上运行 t-SNE 可视化的两个样例：

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

## 如何可视化自监督模型的t-SNE（如 MAE）

以下是在ImageNet数据集上训练的 MAE-ViT-base 模型上运行 t-SNE 可视化的一个样例。输入数据来自 ImageNet 验证集。MAE和一些自监督预训练算法配置中没有 test_dataloader 信息。在分析这些自监督算法时，你需要在配置中添加 test_dataloader 信息，或者使用 `--test-cfg` 字段来指定一个配置文件。

```shell
python tools/visualization/vis_tsne.py \
    configs/mae/mae_vit-base-p16_8xb512-amp-coslr-800e_in1k.py \
    --checkpoint https://download.openmmlab.com/mmselfsup/1.x/mae/mae_vit-base-p16_8xb512-fp16-coslr-800e_in1k/mae_vit-base-p16_8xb512-coslr-800e-fp16_in1k_20220825-5d81fbc4.pth \
    --test-cfg configs/_base_/datasets/imagenet_bs32.py
```

| MAE-ViT-base                                                                                                                                                  |
| ------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| <div align=center><img src='https://github.com/open-mmlab/mmpretrain/assets/42371271/ee576c0c-abef-43d1-8866-24a5f5fd0cf6' height="auto" width="auto" ></div> |
