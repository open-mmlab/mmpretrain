# Verify Dataset

In MMPretrain, we also provide a tool `tools/misc/verify_dataset.py` to check whether there exists **broken pictures** in the given dataset.

## Introduce the tool

```shell
python tools/print_config.py \
    ${CONFIG} \
    [--out-path ${OUT-PATH}] \
    [--phase ${PHASE}] \
    [--num-process ${NUM-PROCESS}]
    [--cfg-options ${CFG_OPTIONS}]
```

**Description of all arguments**:

- `config` : The path of the model config file.
- `--out-path` : The path to save the verification result, if not set, defaults to 'brokenfiles.log'.
- `--phase` :  Phase of dataset to verify, accept "train" "test" and "val", if not set, defaults to "train".
- `--num-process` : number of process to use, if not set, defaults to 1.
- `--cfg-options`: If specified, the key-value pair config will be merged into the config file, for more details please refer to [Learn about Configs](../user_guides/config.md)

## Example

```shell
python tools/misc/verify_dataset.py configs/t2t_vit/t2t-vit-t-14_8xb64_in1k.py --out-path broken_imgs.log --phase val --num-process 8
```
