# 其他工具

<!-- TOC -->

- [打印完整配置](#打印完整配置)
- [检查数据集](#检查数据集)
- [常见问题](#常见问题)

<!-- TOC -->

## 打印完整配置

`tools/misc/print_config.py` 脚本会解析所有输入变量，并打印完整配置信息。

```shell
python tools/misc/print_config.py ${CONFIG} [--cfg-options ${CFG_OPTIONS}]
```

**所有参数说明**：

- `config` ：配置文件的路径。
- `--cfg-options`: 额外的配置选项，会被合入配置文件，参考[教程 1：如何编写配置文件](https://mmclassification.readthedocs.io/zh_CN/latest/tutorials/config.html)。

**示例**：

```shell
python tools/misc/print_config.py configs/t2t_vit/t2t-vit-t-14_8xb64_in1k.py
```

## 检查数据集

`tools/misc/verify_dataset.py` 脚本会检查数据集的所有图片，查看是否有已经损坏的图片。

```shell
python tools/print_config.py \
    ${CONFIG} \
    [--out-path ${OUT-PATH}] \
    [--phase ${PHASE}] \
    [--num-process ${NUM-PROCESS}]
    [--cfg-options ${CFG_OPTIONS}]
```

**所有参数说明**:

- `config` ： 配置文件的路径。
- `--out-path` ： 输出结果路径，默认为 'brokenfiles.log'。
- `--phase` ： 检查哪个阶段的数据集，可用值为 "train" 、"test" 或者 "val"， 默认为 "train"。
- `--num-process` ： 指定的进程数，默认为1。
- `--cfg-options`: 额外的配置选项，会被合入配置文件，参考[教程 1：如何编写配置文件](https://mmclassification.readthedocs.io/zh_CN/latest/tutorials/config.html)。

**示例**:

```shell
python tools/misc/verify_dataset.py configs/t2t_vit/t2t-vit-t-14_8xb64_in1k.py --out-path broken_imgs.log --phase val --num-process 8
```

## 常见问题

- 无
