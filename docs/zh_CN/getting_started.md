# 基础教程

本文档提供 MMClassification 相关用法的基本教程。

## 准备数据集

MMClassification 建议用户将数据集根目录链接到 `$MMCLASSIFICATION/data` 下。
如果用户的文件夹结构与默认结构不同，则需要在配置文件中进行对应路径的修改。

```
mmclassification
├── mmcls
├── tools
├── configs
├── docs
├── data
│   ├── imagenet
│   │   ├── meta
│   │   ├── train
│   │   ├── val
│   ├── cifar
│   │   ├── cifar-10-batches-py
│   ├── mnist
│   │   ├── train-images-idx3-ubyte
│   │   ├── train-labels-idx1-ubyte
│   │   ├── t10k-images-idx3-ubyte
│   │   ├── t10k-labels-idx1-ubyte

```

对于 ImageNet，其存在多个版本，但最为常用的一个是 [ILSVRC 2012](http://www.image-net.org/challenges/LSVRC/2012/)，可以通过以下步骤获取该数据集。

1. 注册账号并登录 [下载页面](http://www.image-net.org/download-images)
2. 获取 ILSVRC2012 下载链接并下载以下文件
   - ILSVRC2012_img_train.tar (~138GB)
   - ILSVRC2012_img_val.tar (~6.3GB)
3. 解压下载的文件
4. 使用 [该脚本](https://github.com/BVLC/caffe/blob/master/data/ilsvrc12/get_ilsvrc_aux.sh) 获取元数据

对于 MNIST，CIFAR10 和 CIFAR100，程序将会在需要的时候自动下载数据集。

对于用户自定义数据集的准备，请参阅 [教程 3：如何自定义数据集
](tutorials/new_dataset.md)

## 使用预训练模型进行推理

MMClassification 提供了一些脚本用于进行单张图像的推理、数据集的推理和数据集的测试（如 ImageNet 等）

### 单张图像的推理

```shell
python demo/image_demo.py ${IMAGE_FILE} ${CONFIG_FILE} ${CHECKPOINT_FILE}

# Example
python demo/image_demo.py demo/demo.JPEG configs/resnet/resnet50_8xb32_in1k.py \
  https://download.openmmlab.com/mmclassification/v0/resnet/resnet50_8xb32_in1k_20210831-ea4938fc.pth
```

### 数据集的推理与测试

- 支持单 GPU
- 支持 CPU
- 支持单节点多 GPU
- 支持多节点

用户可使用以下命令进行数据集的推理：

```shell
# 单 GPU
python tools/test.py ${CONFIG_FILE} ${CHECKPOINT_FILE} [--metrics ${METRICS}] [--out ${RESULT_FILE}]

# CPU: 禁用 GPU 并运行单 GPU 测试脚本
export CUDA_VISIBLE_DEVICES=-1
python tools/test.py ${CONFIG_FILE} ${CHECKPOINT_FILE} [--metrics ${METRICS}] [--out ${RESULT_FILE}]

# 多 GPU
./tools/dist_test.sh ${CONFIG_FILE} ${CHECKPOINT_FILE} ${GPU_NUM} [--metrics ${METRICS}] [--out ${RESULT_FILE}]

# 基于 slurm 分布式环境的多节点
python tools/test.py ${CONFIG_FILE} ${CHECKPOINT_FILE} [--metrics ${METRICS}] [--out ${RESULT_FILE}] --launcher slurm
```

可选参数：

- `RESULT_FILE`：输出结果的文件名。如果未指定，结果将不会保存到文件中。支持 json, yaml, pickle 格式。
- `METRICS`：数据集测试指标，如准确率 (accuracy), 精确率 (precision), 召回率 (recall) 等

例子：

在 ImageNet 验证集上，使用 ResNet-50 进行推理并获得预测标签及其对应的预测得分。

```shell
python tools/test.py configs/resnet/resnet50_8xb16_cifar10.py \
  https://download.openmmlab.com/mmclassification/v0/resnet/resnet50_b16x8_cifar10_20210528-f54bfad9.pth \
  --out result.pkl
```

## 模型训练

MMClassification 使用 `MMDistributedDataParallel` 进行分布式训练，使用 `MMDataParallel` 进行非分布式训练。

所有的输出（日志文件和模型权重文件）会被将保存到工作目录下。工作目录通过配置文件中的参数 `work_dir` 指定。

默认情况下，MMClassification 在每个周期后会在验证集上评估模型，可以通过在训练配置中修改 `interval` 参数来更改评估间隔

```python
evaluation = dict(interval=12)  # 每进行 12 轮训练后评估一次模型
```

### 使用单个 GPU 进行训练

```shell
python tools/train.py ${CONFIG_FILE} [optional arguments]
```

如果用户想在命令中指定工作目录，则需要增加参数 `--work-dir ${YOUR_WORK_DIR}`

### 使用 CPU 训练

使用 CPU 训练的流程和使用单 GPU 训练的流程一致，我们仅需要在训练流程开始前禁用 GPU。

```shell
export CUDA_VISIBLE_DEVICES=-1
```

之后运行单 GPU 训练脚本即可。

```{warning}
我们不推荐用户使用 CPU 进行训练，这太过缓慢。我们支持这个功能是为了方便用户在没有 GPU 的机器上进行调试。
```

### 使用单台机器多个 GPU 进行训练

```shell
./tools/dist_train.sh ${CONFIG_FILE} ${GPU_NUM} [optional arguments]
```

可选参数为：

- `--no-validate` (**不建议**): 默认情况下，程序将会在训练期间的每 k （默认为 1) 个周期进行一次验证。要禁用这一功能，使用 `--no-validate`
- `--work-dir ${WORK_DIR}`：覆盖配置文件中指定的工作目录。
- `--resume-from ${CHECKPOINT_FILE}`：从以前的模型权重文件恢复训练。

`resume-from` 和 `load-from` 的不同点：
`resume-from` 加载模型参数和优化器状态，并且保留检查点所在的周期数，常被用于恢复意外被中断的训练。
`load-from` 只加载模型参数，但周期数从 0 开始计数，常被用于微调模型。

### 使用多台机器进行训练

如果您想使用由 ethernet 连接起来的多台机器， 您可以使用以下命令:

在第一台机器上:

```shell
NNODES=2 NODE_RANK=0 PORT=$MASTER_PORT MASTER_ADDR=$MASTER_ADDR sh tools/dist_train.sh $CONFIG $GPUS
```

在第二台机器上:

```shell
NNODES=2 NODE_RANK=1 PORT=$MASTER_PORT MASTER_ADDR=$MASTER_ADDR sh tools/dist_train.sh $CONFIG $GPUS
```

但是，如果您不使用高速网路连接这几台机器的话，训练将会非常慢。

如果用户在 [slurm](https://slurm.schedmd.com/) 集群上运行 MMClassification，可使用 `slurm_train.sh` 脚本。（该脚本也支持单台机器上进行训练）

```shell
[GPUS=${GPUS}] ./tools/slurm_train.sh ${PARTITION} ${JOB_NAME} ${CONFIG_FILE} ${WORK_DIR}
```

用户可以在 [slurm_train.sh](https://github.com/open-mmlab/mmclassification/blob/master/tools/slurm_train.sh) 中检查所有的参数和环境变量

如果用户的多台机器通过 Ethernet 连接，则可以参考 pytorch [launch utility](https://pytorch.org/docs/stable/distributed.html#launch-utility)。如果用户没有高速网络，如 InfiniBand，速度将会非常慢。

### 使用单台机器启动多个任务

如果用使用单台机器启动多个任务，如在有 8 块 GPU 的单台机器上启动 2 个需要 4 块 GPU 的训练任务，则需要为每个任务指定不同端口，以避免通信冲突。

如果用户使用 `dist_train.sh` 脚本启动训练任务，则可以通过以下命令指定端口

```shell
CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=29500 ./tools/dist_train.sh ${CONFIG_FILE} 4
CUDA_VISIBLE_DEVICES=4,5,6,7 PORT=29501 ./tools/dist_train.sh ${CONFIG_FILE} 4
```

如果用户在 slurm 集群下启动多个训练任务，则需要修改配置文件中的 `dist_params` 变量，以设置不同的通信端口。

在 `config1.py` 中，

```python
dist_params = dict(backend='nccl', port=29500)
```

在 `config2.py` 中，

```python
dist_params = dict(backend='nccl', port=29501)
```

之后便可启动两个任务，分别对应 `config1.py` 和 `config2.py`。

```shell
CUDA_VISIBLE_DEVICES=0,1,2,3 GPUS=4 ./tools/slurm_train.sh ${PARTITION} ${JOB_NAME} config1.py ${WORK_DIR}
CUDA_VISIBLE_DEVICES=4,5,6,7 GPUS=4 ./tools/slurm_train.sh ${PARTITION} ${JOB_NAME} config2.py ${WORK_DIR}
```

## 实用工具

我们在 `tools/` 目录下提供的一些对训练和测试十分有用的工具

### 计算 FLOPs 和参数量（试验性的）

我们根据 [flops-counter.pytorch](https://github.com/sovrasov/flops-counter.pytorch) 提供了一个脚本用于计算给定模型的 FLOPs 和参数量

```shell
python tools/analysis_tools/get_flops.py ${CONFIG_FILE} [--shape ${INPUT_SHAPE}]
```

用户将获得如下结果：

```
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

### 模型发布

在发布模型之前，你也许会需要

1. 转换模型权重至 CPU 张量
2. 删除优化器状态
3. 计算模型权重文件的哈希值，并添加至文件名之后

```shell
python tools/convert_models/publish_model.py ${INPUT_FILENAME} ${OUTPUT_FILENAME}
```

例如：

```shell
python tools/convert_models/publish_model.py work_dirs/resnet50/latest.pth imagenet_resnet50.pth
```

最终输出的文件名将会是 `imagenet_resnet50_{date}-{hash id}.pth`

## 详细教程

目前，MMClassification 提供以下几种更详细的教程：

- [如何编写配置文件](tutorials/config.md)
- [如何微调模型](tutorials/finetune.md)
- [如何增加新数据集](tutorials/new_dataset.md)
- [如何设计数据处理流程](tutorials/data_pipeline.md)
- [如何增加新模块](tutorials/new_modules.md)
- [如何自定义优化策略](tutorials/schedule.md)
- [如何自定义运行参数](tutorials/runtime.md)。
