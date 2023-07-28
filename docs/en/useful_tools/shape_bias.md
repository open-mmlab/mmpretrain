# Shape Bias Tool Usage

Shape bias measures how a model relies the shapes, compared to texture, to sense the semantics in images. For more details,
we recommend interested readers to this [paper](https://arxiv.org/abs/2106.07411). MMPretrain provide an off-the-shelf toolbox to
obtain the shape bias of a classification model. You can following these steps below:

## Prepare the dataset

First you should download the [cue-conflict](https://github.com/bethgelab/model-vs-human/releases/download/v0.1/cue-conflict.tar.gz) to `data` folder,
and then unzip this dataset. After that, you `data` folder should have the following structure:

```text
data
├──cue-conflict
|      |──airplane
|      |──bear
|      ...
|      |── truck
```

## Modify the config for classification

We run the shape-bias tool on a ViT-base model with masked autoencoder pretraining. Its config file is `configs/mae/benchmarks/vit-base-p16_8xb128-coslr-100e_in1k.py`, and its checkpoint is downloaded from [this link](https://download.openmmlab.com/mmselfsup/1.x/mae/mae_vit-base-p16_8xb512-fp16-coslr-1600e_in1k/vit-base-p16_ft-8xb128-coslr-100e_in1k/vit-base-p16_ft-8xb128-coslr-100e_in1k_20220825-cf70aa21.pth). Replace the original test_pipeline, test_dataloader and test_evaluation with the following configurations:

```python
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='ResizeEdge',
        scale=256,
        edge='short',
        backend='pillow'),
    dict(type='CenterCrop', crop_size=224),
    dict(type='PackInputs')
]
test_dataloader = dict(
    pin_memory=True,
    collate_fn=dict(type='default_collate'),
    batch_size=32,
    num_workers=4,
    dataset=dict(
        type='CustomDataset',
        data_root='data/cue-conflict',
        pipeline=test_pipeline,
        _delete_=True),
    sampler=dict(type='DefaultSampler', shuffle=False),
    persistent_workers=True,
    drop_last=False)
test_evaluator = dict(
    type='mmpretrain.ShapeBiasMetric',
    _delete_=True,
    csv_dir='work_dirs/shape_bias',
    model_name='mae')
```

Please note you should make custom modifications to the `csv_dir` and `model_name` above. I renamed my modified sample config file as `vit-base-p16_8xb128-coslr-100e_in1k_shape-bias.py` in the folder `configs/mae/benchmarks/`.

## Inference your model with above modified config file

Then you should inferece your model on the `cue-conflict` dataset with the your modified config file.

```shell
# For PyTorch
bash tools/dist_test.sh $CONFIG $CHECKPOINT
```

**Description of all arguments**:

- `$CONFIG`: The path of your modified config file.
- `$CHECKPOINT`: The path or link of the checkpoint file.

```shell
# Example
bash tools/dist_test.sh configs/mae/benchmarks/vit-base-p16_8xb128-coslr-100e_in1k_shape-bias.py https://download.openmmlab.com/mmselfsup/1.x/mae/mae_vit-base-p16_8xb512-fp16-coslr-1600e_in1k/vit-base-p16_ft-8xb128-coslr-100e_in1k/vit-base-p16_ft-8xb128-coslr-100e_in1k_20220825-cf70aa21.pth 1
```

After that, you should obtain a csv file in `csv_dir` folder, named `cue-conflict_model-name_session-1.csv`. Besides this file, you should also download these [csv files](https://github.com/bethgelab/model-vs-human/tree/master/raw-data/cue-conflict) to the
`csv_dir`.

## Plot shape bias

Then we can start to plot the shape bias:

```shell
python tools/analysis_tools/shape_bias.py --csv-dir $CSV_DIR --result-dir $RESULT_DIR --colors $RGB --markers o --plotting-names $YOUR_MODEL_NAME --model-names $YOUR_MODEL_NAME
```

**Description of all arguments**:

- `--csv-dir $CSV_DIR`, the same directory to save these csv files.
- `--result-dir $RESULT_DIR`, the directory to output the result named `cue-conflict_shape-bias_matrixplot.pdf`.
- `--colors $RGB`, should be the RGB values, formatted in R G B, e.g. 100 100 100, and can be multiple RGB values, if you want to plot the shape bias of several models.
- `--plotting-names $YOUR_MODEL_NAME`, the name of the legend in the shape bias figure, and you can set it as your model name. If you want to plot several models, plotting_names can be multiple values.
- `model-names $YOUR_MODEL_NAME`, should be the same name specified in your config, and can be multiple names if you want to plot the shape bias of several models.

Please note, every three values for `--colors` corresponds to one value for `--model-names`. After all of above steps, you are expected to obtain the following figure.

<div align="center">
<img src="https://github.com/open-mmlab/mmpretrain/assets/42371271/dc608d06-43eb-4860-bb70-486ed2a3f927" width="500" />
</div>
