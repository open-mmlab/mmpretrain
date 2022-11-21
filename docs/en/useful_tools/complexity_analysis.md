# Model Complexity Analysis

## Get the FLOPs and params (experimental)

We provide a script adapted from [fvcore](https://github.com/facebookresearch/fvcore/blob/main/fvcore/nn/flop_count.py) to compute the FLOPs and params of a given model.

```shell
python tools/analysis_tools/get_flops.py ${CONFIG_FILE} [--shape ${INPUT_SHAPE}]
```

Description of all arguments:

- `config` : The path of the model config file.
- `--shape`: Input size, support single value or double value parameter, such as `--shape 256` or `--shape 224 256`. If not set, default to be `224 224`.

You will get the final result like this.

```text
==============================
Input shape: (3, 224, 224)
Flops: 17.582G
Params: 91.234M
Activation: 23.895M
==============================
```

Also, you will get the detailed complexity information of each layer like this:

```text
| module                                    | #parameters or shape   | #flops    | #activations   |
|:------------------------------------------|:-----------------------|:----------|:---------------|
| model                                     | 91.234M                | 17.582G   | 23.895M        |
|  backbone                                 |  85.799M               |  17.582G  |  23.895M       |
|   backbone.cls_token                      |   (1, 1, 768)          |           |                |
|   backbone.pos_embed                      |   (1, 197, 768)        |           |                |
|   backbone.patch_embed.projection         |   0.591M               |   0.116G  |   0.151M       |
|    backbone.patch_embed.projection.weight |    (768, 3, 16, 16)    |           |                |
|    backbone.patch_embed.projection.bias   |    (768,)              |           |                |
|   backbone.layers                         |   85.054M              |   17.466G |   23.744M      |
|    backbone.layers.0                      |    7.088M              |    1.455G |    1.979M      |
|    backbone.layers.1                      |    7.088M              |    1.455G |    1.979M      |
|    backbone.layers.2                      |    7.088M              |    1.455G |    1.979M      |
|    backbone.layers.3                      |    7.088M              |    1.455G |    1.979M      |
|    backbone.layers.4                      |    7.088M              |    1.455G |    1.979M      |
|    backbone.layers.5                      |    7.088M              |    1.455G |    1.979M      |
|    backbone.layers.6                      |    7.088M              |    1.455G |    1.979M      |
|    backbone.layers.7                      |    7.088M              |    1.455G |    1.979M      |
|    backbone.layers.8                      |    7.088M              |    1.455G |    1.979M      |
|    backbone.layers.9                      |    7.088M              |    1.455G |    1.979M      |
|    backbone.layers.10                     |    7.088M              |    1.455G |    1.979M      |
|    backbone.layers.11                     |    7.088M              |    1.455G |    1.979M      |
|   backbone.ln1                            |   1.536K               |   0.756M  |   0            |
|    backbone.ln1.weight                    |    (768,)              |           |                |
|    backbone.ln1.bias                      |    (768,)              |           |                |
|  head.layers                              |  5.435M                |           |                |
|   head.layers.pre_logits                  |   2.362M               |           |                |
|    head.layers.pre_logits.weight          |    (3072, 768)         |           |                |
|    head.layers.pre_logits.bias            |    (3072,)             |           |                |
|   head.layers.head                        |   3.073M               |           |                |
|    head.layers.head.weight                |    (1000, 3072)        |           |                |
|    head.layers.head.bias                  |    (1000,)             |           |                |
```

```{warning}
This tool is still experimental and we do not guarantee that the number is correct. You may well use the result for simple comparisons, but double-check it before you adopt it in technical reports or papers.
- FLOPs are related to the input shape while parameters are not. The default input shape is (1, 3, 224, 224).
- Some operators are not counted into FLOPs like custom operators. Refer to [`fvcore.nn.flop_count._DEFAULT_SUPPORTED_OPS`](https://github.com/facebookresearch/fvcore/blob/main/fvcore/nn/flop_count.py) for details.
```
