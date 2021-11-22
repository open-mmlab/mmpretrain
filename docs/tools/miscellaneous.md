# MISCELLANEOUS

<!-- TOC -->

- [Print the entire config](#print-the-entire-config)
- [Verify Dataset](#verify-dataset)
- [FAQs](#faqs)

<!-- TOC -->

## Print the entire config

`tools/misc/print_config.py` prints the whole config verbatim, expanding all its imports.

```shell
python tools/misc/print_config.py ${CONFIG} [--cfg-options ${CFG_OPTIONS}]
```

Description of all arguments:

- `config` : The path of a model config file.
- `--cfg-options`: If specified, the key-value pair optional cfg will be merged into config file, for more details please refer to [Tutorial 1: Learn about Configs](../tutorials/config.md)

**Examples**:

```shell
python tools/misc/print_config.py configs/t2t_vit/t2t-vit-t-14_8xb64_in1k.py
```

## Verify Dataset

`tools/misc/verify_dataset.py` can verify dataset, check whether there is broken picture in given dataset.

```shell
python tools/print_config.py \
    ${CONFIG} \
    [--out-path ${OUT-PATH}] \
    [--phase ${PHASE}] \
    [--num-process ${NUM-PROCESS}]
    [--cfg-options ${CFG_OPTIONS}]
```

**Description of all arguments**:

- `config` : The path of a model config file.
- `--out-path` : The path to save the verification result, if not set, default to be 'brokenfiles.log'.
- `--phase` :  Phase of dataset to verify, accept "train" "test" and "val", if not set, default to be "train".
- `--num-process` : number of process to use, if not set, default to be 1.
- `--cfg-options`: If specified, the key-value pair optional cfg will be merged into config file, for more details please refer to [Tutorial 1: Learn about Configs](../tutorials/config.md)

**Examples**:

```shell
python tools/analysis/print_config.py configs/t2t_vit/t2t-vit-t-14_8xb64_in1k.py --out-path broken_imgs.log --phase val --num-process 8
```

## FAQs

- None
