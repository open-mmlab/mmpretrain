# 测试

## 单机单卡测试

你可以使用 `tools/test.py` 在电脑上用 CPU 或是 GPU 进行模型的测试。

以下是测试脚本的完整用法：

```shell
python tools/test.py ${CONFIG_FILE} ${CHECKPOINT_FILE} [ARGS]
```

````{note}
默认情况下，MMPretrain 会自动调用你的 GPU 进行测试。如果你有 GPU 但仍想使用 CPU 进行测试，请设置环境变量 `CUDA_VISIBLE_DEVICES` 为空或者 -1 来对禁用 GPU。

```bash
CUDA_VISIBLE_DEVICES=-1 python tools/test.py ${CONFIG_FILE} ${CHECKPOINT_FILE} [ARGS]
```
````

| 参数                                  | 描述                                                                                                                                                                |
| ------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `CONFIG_FILE`                         | 配置文件的路径。                                                                                                                                                    |
| `CHECKPOINT_FILE`                     | 权重文件路径（支持 http 链接，你可以在[这里](https://mmpretrain.readthedocs.io/en/latest/modelzoo_statistics.html)寻找需要的权重文件）。                            |
| `--work-dir WORK_DIR`                 | 用来保存测试指标结果的文件夹。                                                                                                                                      |
| `--out OUT`                           | 用来保存测试输出的文件。                                                                                                                                            |
| `--out-item OUT_ITEM`                 | 指定测试输出文件的内容，可以为 "pred" 或 "metrics"，其中 "pred" 表示保存所有模型输出，这些数据可以用于离线测评；"metrics" 表示输出测试指标。默认为 "pred"。         |
| `--cfg-options CFG_OPTIONS`           | 重载配置文件中的一些设置。使用类似 `xxx=yyy` 的键值对形式指定，这些设置会被融合入从配置文件读取的配置。你可以使用 `key="[a,b]"` 或者 `key=a,b` 的格式来指定列表格式的值，且支持嵌套，例如 \`key="[(a,b),(c,d)]"，这里的引号是不可省略的。另外每个重载项内部不可出现空格。 |
| `--show-dir SHOW_DIR`                 | 用于保存可视化预测结果图像的文件夹。                                                                                                                                |
| `--show`                              | 在窗口中显示预测结果图像。                                                                                                                                          |
| `--interval INTERVAL`                 | 每隔多少样本进行一次预测结果可视化。                                                                                                                                |
| `--wait-time WAIT_TIME`               | 每个窗口的显示时间（单位为秒）。                                                                                                                                    |
| `--no-pin-memory`                     | 是否在 dataloaders 中关闭 `pin_memory` 选项                                                                                                                         |
| `--tta`                               | 是否开启 Test-Time-Aug (TTA). 如果配置文件有 `tta_pipeline` 和 `tta_model`，将使用这些配置指定 TTA transforms，并且决定如何融合 TTA 的结果。 否则，通过平均分类分数使用 flip TTA。 |
| `--launcher {none,pytorch,slurm,mpi}` | 启动器，默认为 "none"。                                                                                                                                             |

## 单机多卡测试

我们提供了一个 shell 脚本，可以使用 `torch.distributed.launch` 启动多 GPU 任务。

```shell
bash ./tools/dist_test.sh ${CONFIG_FILE} ${CHECKPOINT_FILE} ${GPU_NUM} [PY_ARGS]
```

| 参数              | 描述                                                                                                                                     |
| ----------------- | ---------------------------------------------------------------------------------------------------------------------------------------- |
| `CONFIG_FILE`     | 配置文件的路径。                                                                                                                         |
| `CHECKPOINT_FILE` | 权重文件路径（支持 http 链接，你可以在[这里](https://mmpretrain.readthedocs.io/en/latest/modelzoo_statistics.html)寻找需要的权重文件）。 |
| `GPU_NUM`         | 使用的 GPU 数量。                                                                                                                        |
| `[PY_ARGS]`       | `tools/test.py` 支持的其他可选参数，参见[上文](#单机单卡测试)。                                                                          |

你还可以使用环境变量来指定启动器的额外参数，比如用如下命令将启动器的通讯端口变更为 29666：

```shell
PORT=29666 bash ./tools/dist_test.sh ${CONFIG_FILE} ${CHECKPOINT_FILE} ${GPU_NUM} [PY_ARGS]
```

如果你希望使用不同的 GPU 进行多项测试任务，可以在启动时指定不同的通讯端口和不同的可用设备。

```shell
CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=29500 bash ./tools/dist_test.sh ${CONFIG_FILE1} ${CHECKPOINT_FILE} 4 [PY_ARGS]
CUDA_VISIBLE_DEVICES=4,5,6,7 PORT=29501 bash ./tools/dist_test.sh ${CONFIG_FILE2} ${CHECKPOINT_FILE} 4 [PY_ARGS]
```

## 多机测试

### 同一网络下的多机

如果你希望使用同一局域网下连接的多台电脑进行一个测试任务，可以使用如下命令：

在第一台机器上：

```shell
NNODES=2 NODE_RANK=0 PORT=$MASTER_PORT MASTER_ADDR=$MASTER_ADDR bash tools/dist_test.sh $CONFIG $CHECKPOINT_FILE $GPUS
```

在第二台机器上：

```shell
NNODES=2 NODE_RANK=1 PORT=$MASTER_PORT MASTER_ADDR=$MASTER_ADDR bash tools/dist_test.sh $CONFIG $CHECKPOINT_FILE $GPUS
```

和单机多卡相比，你需要指定一些额外的环境变量：

| 环境变量      | 描述                                           |
| ------------- | ---------------------------------------------- |
| `NNODES`      | 机器总数。                                     |
| `NODE_RANK`   | 本机的序号                                     |
| `PORT`        | 通讯端口，它在所有机器上都应当是一致的。       |
| `MASTER_ADDR` | 主机的 IP 地址，它在所有机器上都应当是一致的。 |

### Slurm 管理下的多机集群

如果你在 [slurm](https://slurm.schedmd.com/) 集群上，可以使用 `tools/slurm_test.sh` 脚本启动任务。

```shell
[ENV_VARS] ./tools/slurm_test.sh ${PARTITION} ${JOB_NAME} ${CONFIG_FILE} ${CHECKPOINT_FILE} [PY_ARGS]
```

这里是该脚本的一些参数：

| 参数              | 描述                                                                                                                                     |
| ----------------- | ---------------------------------------------------------------------------------------------------------------------------------------- |
| `PARTITION`       | 使用的集群分区。                                                                                                                         |
| `JOB_NAME`        | 任务的名称，你可以随意起一个名字。                                                                                                       |
| `CONFIG_FILE`     | 配置文件路径。                                                                                                                           |
| `CHECKPOINT_FILE` | 权重文件路径（支持 http 链接，你可以在[这里](https://mmpretrain.readthedocs.io/en/latest/modelzoo_statistics.html)寻找需要的权重文件）。 |
| `[PY_ARGS]`       | `tools/test.py` 支持的其他可选参数，参见[上文](#单机单卡测试)。                                                                          |

这里是一些你可以用来配置 slurm 任务的环境变量：

| 环境变量        | 描述                                                                                       |
| --------------- | ------------------------------------------------------------------------------------------ |
| `GPUS`          | 使用的 GPU 总数，默认为 8。                                                                |
| `GPUS_PER_NODE` | 每个节点分配的 GPU 数，你可以根据节点情况指定。默认为 8。                                  |
| `CPUS_PER_TASK` | 每个任务分配的 CPU 数（通常一个 GPU 对应一个任务）。默认为 5。                             |
| `SRUN_ARGS`     | `srun` 命令支持的其他参数。可用的选项参见[官方文档](https://slurm.schedmd.com/srun.html)。 |
