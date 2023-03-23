# 打印完整配置文件

`print_config.py`脚本脚本会解析所有输入变量，并打印完整配置信息。

## 说明：

`tools/misc/print_config.py` 脚本会逐字打印整个配置文件，并展示所有导入的文件。

```shell
python tools/misc/print_config.py ${CONFIG} [--cfg-options ${CFG_OPTIONS}]
```

所有参数的说明：

- `config`：模型配置文件的路径。
- `--cfg-options`：额外的配置选项，会被合入配置文件，参考[学习配置文件](../user_guides/config.md)。

## 示例：

打印`configs/t2t_vit/t2t-vit-t-14_8xb64_in1k.py`文件的完整配置

```shell
# 打印完成的配置文件
python tools/misc/print_config.py configs/t2t_vit/t2t-vit-t-14_8xb64_in1k.py

# 将完整的配置文件保存为一个独立的配置文件
python tools/misc/print_config.py configs/t2t_vit/t2t-vit-t-14_8xb64_in1k.py > final_config.py
```
