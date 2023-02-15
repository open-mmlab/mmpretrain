# How to Get the Complete Config

We also provide the `print_config.py` tools to print the complete configuration of the given experiment.
You can check each item of the config before the training by using the following command.

## Description

`tools/misc/print_config.py` prints the whole config verbatim, expanding all its imports.

```shell
python tools/misc/print_config.py ${CONFIG} [--cfg-options ${CFG_OPTIONS}]
```

Description of all arguments:

- `config` : The path of the model config file.
- `--cfg-options`: If specified, the key-value pair config will be merged into the config file, for more details please refer to [Learn about Configs](../user_guides/config.md)

## Examples

```shell
# Print a complete config
python tools/misc/print_config.py configs/t2t_vit/t2t-vit-t-14_8xb64_in1k.py

# Save the complete config to a independent config file.
python tools/misc/print_config.py configs/t2t_vit/t2t-vit-t-14_8xb64_in1k.py > final_config.py
```
