# 训练

在本教程中，我们将介绍如何使用 MMPretrain 中提供的脚本启动训练任务。
如果你需要了解一些具体的训练例子，可以查阅 [如何在自定义数据集上进行模型预训练](../notes/pretrain_custom_dataset.md) 和 [如何在自定义数据集上微调模型](../notes/finetune_custom_dataset.md).

## 单机单卡训练

你可以使用 `tools/train.py` 在电脑上用 CPU 或是 GPU 进行模型的训练。

以下是训练脚本的完整用法：

```shell
python tools/train.py ${CONFIG_FILE} [ARGS]
```

````{note}
默认情况下，MMPretrain 会自动调用你的 GPU 进行训练。如果你有 GPU 但仍想使用 CPU 进行训练，请设置环境变量 `CUDA_VISIBLE_DEVICES` 为空或者 -1 来对禁用 GPU。

```bash
CUDA_VISIBLE_DEVICES=-1 python tools/train.py ${CONFIG_FILE} [ARGS]
```
````

| 参数                                  | 描述                                                                                                                                                                |
| ------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `CONFIG_FILE`                         | 配置文件的路径。                                                                                                                                                    |
| `--work-dir WORK_DIR`                 | 用来保存训练日志和权重文件的文件夹，默认是 `./work_dirs` 目录下，与配置文件同名的文件夹。                                                                           |
| `--resume [RESUME]`                   | 恢复训练。如果指定了权重文件路径，则从指定的权重文件恢复；如果没有指定，则尝试从最新的权重文件进行恢复。                                                            |
| `--amp`                               | 启用混合精度训练。                                                                                                                                                  |
| `--no-validate`                       | **不建议** 在训练过程中不进行验证集上的精度验证。                                                                                                                   |
| `--auto-scale-lr`                     | 自动根据实际的批次大小（batch size）和预设的批次大小对学习率进行缩放。                                                                                              |
| `--no-pin-memory`                     | 是否在 dataloaders 中关闭 `pin_memory` 选项                                                                                                                         |
| `--no-persistent-workers`             | 是否在 dataloaders 中关闭 `persistent_workers` 选项                                                                                                                 |
| `--cfg-options CFG_OPTIONS`           | 重载配置文件中的一些设置。使用类似 `xxx=yyy` 的键值对形式指定，这些设置会被融合入从配置文件读取的配置。你可以使用 `key="[a,b]"` 或者 `key=a,b` 的格式来指定列表格式的值，且支持嵌套，例如 \`key="[(a,b),(c,d)]"，这里的引号是不可省略的。另外每个重载项内部不可出现空格。 |
| `--launcher {none,pytorch,slurm,mpi}` | 启动器，默认为 "none"。                                                                                                                                             |

## 单机多卡训练

我们提供了一个 shell 脚本，可以使用 `torch.distributed.launch` 启动多 GPU 任务。

```shell
bash ./tools/dist_train.sh ${CONFIG_FILE} ${GPU_NUM} [PY_ARGS]
```

| 参数          | 描述                                                             |
| ------------- | ---------------------------------------------------------------- |
| `CONFIG_FILE` | 配置文件的路径。                                                 |
| `GPU_NUM`     | 使用的 GPU 数量。                                                |
| `[PY_ARGS]`   | `tools/train.py` 支持的其他可选参数，参见[上文](#单机单卡训练)。 |

你还可以使用环境变量来指定启动器的额外参数，比如用如下命令将启动器的通讯端口变更为 29666：

```shell
PORT=29666 bash ./tools/dist_train.sh ${CONFIG_FILE} ${GPU_NUM} [PY_ARGS]
```

如果你希望使用不同的 GPU 进行多项训练任务，可以在启动时指定不同的通讯端口和不同的可用设备。

```shell
CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=29500 bash ./tools/dist_train.sh ${CONFIG_FILE1} 4 [PY_ARGS]
CUDA_VISIBLE_DEVICES=4,5,6,7 PORT=29501 bash ./tools/dist_train.sh ${CONFIG_FILE2} 4 [PY_ARGS]
```

## 多机训练

### 同一网络下的多机

如果你希望使用同一局域网下连接的多台电脑进行一个训练任务，可以使用如下命令：

在第一台机器上：

```shell
NNODES=2 NODE_RANK=0 PORT=$MASTER_PORT MASTER_ADDR=$MASTER_ADDR bash tools/dist_train.sh $CONFIG $GPUS
```

在第二台机器上：

```shell
NNODES=2 NODE_RANK=1 PORT=$MASTER_PORT MASTER_ADDR=$MASTER_ADDR bash tools/dist_train.sh $CONFIG $GPUS
```

和单机多卡相比，你需要指定一些额外的环境变量：

| 环境变量      | 描述                                           |
| ------------- | ---------------------------------------------- |
| `NNODES`      | 机器总数。                                     |
| `NODE_RANK`   | 本机的序号                                     |
| `PORT`        | 通讯端口，它在所有机器上都应当是一致的。       |
| `MASTER_ADDR` | 主机的 IP 地址，它在所有机器上都应当是一致的。 |

通常来说，如果这几台机器之间不是高速网络连接，训练速度会非常慢。

### Slurm 管理下的多机集群

如果你在 [slurm](https://slurm.schedmd.com/) 集群上，可以使用 `tools/slurm_train.sh` 脚本启动任务。

```shell
[ENV_VARS] ./tools/slurm_train.sh ${PARTITION} ${JOB_NAME} ${CONFIG_FILE} ${WORK_DIR} [PY_ARGS]
```

这里是该脚本的一些参数：

| 参数          | 描述                                                             |
| ------------- | ---------------------------------------------------------------- |
| `PARTITION`   | 使用的集群分区。                                                 |
| `JOB_NAME`    | 任务的名称，你可以随意起一个名字。                               |
| `CONFIG_FILE` | 配置文件路径。                                                   |
| `WORK_DIR`    | 用以保存日志和权重文件的文件夹。                                 |
| `[PY_ARGS]`   | `tools/train.py` 支持的其他可选参数，参见[上文](#单机单卡训练)。 |

这里是一些你可以用来配置 slurm 任务的环境变量：

| 环境变量        | 描述                                                                                       |
| --------------- | ------------------------------------------------------------------------------------------ |
| `GPUS`          | 使用的 GPU 总数，默认为 8。                                                                |
| `GPUS_PER_NODE` | 每个节点分配的 GPU 数，你可以根据节点情况指定。默认为 8。                                  |
| `CPUS_PER_TASK` | 每个任务分配的 CPU 数（通常一个 GPU 对应一个任务）。默认为 5。                             |
| `SRUN_ARGS`     | `srun` 命令支持的其他参数。可用的选项参见[官方文档](https://slurm.schedmd.com/srun.html)。 |
