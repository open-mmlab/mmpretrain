# MaskFeat Pre-training with Video

- [MaskFeat Pre-training with Video](#maskfeat-pre-training-with-video)
  - [Description](#description)
  - [Usage](#usage)
    - [Setup Environment](#setup-environment)
    - [Data Preparation](#data-preparation)
    - [Pre-training Commands](#pre-training-commands)
      - [On Local Single GPU](#on-local-single-gpu)
      - [On Multiple GPUs](#on-multiple-gpus)
      - [On Multiple GPUs with Slurm](#on-multiple-gpus-with-slurm)
    - [Downstream Tasks Commands](#downstream-tasks-commands)
      - [On Multiple GPUs](#on-multiple-gpus-1)
      - [On Multiple GPUs with Slurm](#on-multiple-gpus-with-slurm-1)
  - [Results](#results)
  - [Citation](#citation)
  - [Checklist](#checklist)

## Description

<!-- Share any information you would like others to know. For example:
Author: @xxx.
This is an implementation of \[XXX\]. -->

Author: @fangyixiao18

This is the implementation of **MaskFeat** with video dataset, like Kinetics400.

## Usage

<!-- For a typical model, this section should contain the commands for dataset prepareation, pre-training, downstream tasks. You are also suggested to dump your environment specification to env.yml by `conda env export > env.yml`. -->

### Setup Environment

Requirements:

- MMPretrain >= 1.0.0rc0
- MMAction2 >= 1.0.0rc3

Please refer to [Get Started](https://mmpretrain.readthedocs.io/en/latest/get_started.html) documentation of MMPretrain to finish installation.

Besides, to process the video data, we apply transforms in MMAction2. The instruction to install MMAction2 can be found in [Get Started documentation](https://mmaction2.readthedocs.io/en/1.x/get_started.html).

### Data Preparation

You can refer to the [documentation](https://mmaction2.readthedocs.io/en/1.x/user_guides/2_data_prepare.html) in MMAction2.

### Pre-training Commands

At first, you need to add the current folder to `PYTHONPATH`, so that Python can find your model files. In `projects/maskfeat_video/` root directory, please run command below to add it.

```shell
export PYTHONPATH=`pwd`:$PYTHONPATH
```

Then run the following commands to train the model:

#### On Local Single GPU

```bash
# train with mim
mim train mmpretrain ${CONFIG} --work-dir ${WORK_DIR}

# a specific command example
mim train mmpretrain configs/maskfeat_mvit-small_8xb32-amp-coslr-300e_k400.py \
    --work-dir work_dirs/selfsup/maskfeat_mvit-small_8xb32-amp-coslr-300e_k400/

# train with scripts
python tools/train.py configs/maskfeat_mvit-small_8xb32-amp-coslr-300e_k400.py \
    --work-dir work_dirs/selfsup/maskfeat_mvit-small_8xb32-amp-coslr-300e_k400/
```

#### On Multiple GPUs

```bash
# train with mim
# a specific command examples, 8 GPUs here
mim train mmpretrain configs/maskfeat_mvit-small_8xb32-amp-coslr-300e_k400.py \
    --work-dir work_dirs/selfsup/maskfeat_mvit-small_8xb32-amp-coslr-300e_k400/ \
    --launcher pytorch --gpus 8

# train with scripts
bash tools/dist_train.sh configs/maskfeat_mvit-small_8xb32-amp-coslr-300e_k400.py 8
```

Note:

- CONFIG: the config files under the directory `configs/`
- WORK_DIR: the working directory to save configs, logs, and checkpoints

#### On Multiple GPUs with Slurm

```bash
# train with mim
mim train mmpretrain configs/maskfeat_mvit-small_16xb32-amp-coslr-300e_k400.py \
    --work-dir work_dirs/selfsup/maskfeat_mvit-small_16xb32-amp-coslr-300e_k400/ \
    --launcher slurm --gpus 16 --gpus-per-node 8 \
    --partition ${PARTITION}

# train with scripts
GPUS_PER_NODE=8 GPUS=16 bash tools/slurm_train.sh ${PARTITION} maskfeat-video \
    configs/maskfeat_mvit-small_16xb32-amp-coslr-300e_k400.py \
    --work-dir work_dirs/selfsup/maskfeat_mvit-small_16xb32-amp-coslr-300e_k400/
```

Note:

- CONFIG: the config files under the directory `configs/`
- WORK_DIR: the working directory to save configs, logs, and checkpoints
- PARTITION: the slurm partition you are using

### Downstream Tasks Commands

To evaluate the **MaskFeat MViT** pretrained with MMPretrain, we recommend to run MMAction2:

#### On Multiple GPUs

```bash
# command example for train
mim train mmaction2 ${CONFIG} \
    --work-dir ${WORK_DIR} \
    --launcher pytorch -gpus 8 \
    --cfg-options model.backbone.init_cfg.type=Pretrained \
    model.backbone.init_cfg.checkpoint=${CHECKPOINT} \
    model.backbone.init_cfg.prefix="backbone." \
    ${PY_ARGS}
    [optional args]

mim train mmaction2 configs/mvit-small_ft-8xb8-coslr-100e_k400.py \
    --work-dir work_dirs/benchmarks/maskfeat/training_maskfeat-mvit-k400/ \
    --launcher pytorch -gpus 8 \
    --cfg-options model.backbone.init_cfg.type=Pretrained \
    model.backbone.init_cfg.checkpoint=https://download.openmmlab.com/mmselfsup/1.x/maskfeat/maskfeat_mvit-small_16xb32-amp-coslr-300e_k400/maskfeat_mvit-small_16xb32-amp-coslr-300e_k400_20230131-87d60b6f.pth \
    model.backbone.init_cfg.prefix="backbone." \
    $PY_ARGS

# command example for test
mim test mmaction2 configs/mvit-small_ft-8xb16-coslr-100e_k400.py \
  --checkpoint https://download.openmmlab.com/mmselfsup/1.x/maskfeat/maskfeat_mvit-small_16xb32-amp-coslr-300e_k400/mvit-small_ft-8xb16-coslr-100e_k400/mvit-small_ft-8xb16-coslr-100e_k400_20230131-5e8303f5.pth \
  --work-dir work_dirs/benchmarks/maskfeat/maskfeat-mvit-k400/test/ \
  --launcher pytorch --gpus 8
```

#### On Multiple GPUs with Slurm

```bash
mim train mmaction2 ${CONFIG} \
    --work-dir ${WORK_DIR} \
    --launcher slurm --gpus 8 --gpus-per-node 8 \
    --partition ${PARTITION} \
    --cfg-options model.backbone.init_cfg.type=Pretrained \
    model.backbone.init_cfg.checkpoint=$CHECKPOINT \
    model.backbone.init_cfg.prefix="backbone." \
    $PY_ARGS

mim test mmaction2 ${CONFIG} \
    --checkpoint https://download.openmmlab.com/mmselfsup/1.x/maskfeat/maskfeat_mvit-small_16xb32-amp-coslr-300e_k400/mvit-small_ft-8xb16-coslr-100e_k400/mvit-small_ft-8xb16-coslr-100e_k400_20230131-5e8303f5.pth
    --work-dir ${WORK_DIR} \
    --launcher slurm --gpus 8 --gpus-per-node 8 \
    --partition ${PARTITION} \
    $PY_ARGS
```

Note:

- CONFIG: the config files under the directory `configs/`
- WORK_DIR: the working directory to save configs, logs, and checkpoints
- PARTITION: the slurm partition you are using
- CHECKPOINT: the pretrained checkpoint of MMPretrain saved in working directory, like `$WORK_DIR/epoch_300.pth`
- PY_ARGS: other optional args

## Results

<!-- You should claim whether this is based on the pre-trained weights, which are converted from the official release; or it's a reproduced result obtained from retraining the model in this project. -->

The Fine-tuning results are based on Kinetics400(K400) dataset.

Due to the version of K400 dataset, our pretraining, fine-tuning and the final test results are based on MMAction2 version, which is a little different from PySlowFast version.

<table class="docutils">
<thead>
  <tr>
	    <th>Algorithm</th>
	    <th>Backbone</th>
	    <th>Epoch</th>
      <th>Batch Size</th>
      <th>Fine-tuning</th>
      <th>Pretrain Links</th>
      <th>Fine-tuning Links</th>
	</tr>
  </thead>
  <tbody>
  <tr>
      <td>MaskFeat</td>
	    <td>MViT-small</td>
	    <td>300</td>
      <td>512</td>
      <td>81.8</td>
      <td><a href='https://github.com/open-mmlab/mmselfsup/blob/dev-1.x/projects/maskfeat_video/configs/maskfeat_mvit-small_16xb32-amp-coslr-300e_k400.py'>config</a> | <a href='https://download.openmmlab.com/mmselfsup/1.x/maskfeat/maskfeat_mvit-small_16xb32-amp-coslr-300e_k400/maskfeat_mvit-small_16xb32-amp-coslr-300e_k400_20230131-87d60b6f.pth'>model</a> | <a href='https://download.openmmlab.com/mmselfsup/1.x/maskfeat/maskfeat_mvit-small_16xb32-amp-coslr-300e_k400/maskfeat_mvit-small_16xb32-amp-coslr-300e_k400_20230118_114151.json'>log</a></td>
      <td><a href='https://github.com/open-mmlab/mmselfsup/blob/dev-1.x/projects/maskfeat_video/configs/mvit-small_ft-8xb16-coslr-100e_k400.py'>config</a> | <a href='https://download.openmmlab.com/mmselfsup/1.x/maskfeat/maskfeat_mvit-small_16xb32-amp-coslr-300e_k400/mvit-small_ft-8xb16-coslr-100e_k400/mvit-small_ft-8xb16-coslr-100e_k400_20230131-5e8303f5.pth'>model</a> | <a href='https://download.openmmlab.com/mmselfsup/1.x/maskfeat/maskfeat_mvit-small_16xb32-amp-coslr-300e_k400/mvit-small_ft-8xb16-coslr-100e_k400/mvit-small_ft-8xb16-coslr-100e_k400_20230121_142927.json'>log</a></td>
	</tr>
</tbody>
</table>

Remarks:

- We converted the pretrained model from PySlowFast and run fine-tuning with MMAction2, based on MMAction2 version of K400, we got `81.5` test accuracy. The pretrained model from MMPretrain got `81.8`, as provided above.
- We also tested our model on [other version](https://github.com/facebookresearch/video-nonlocal-net/blob/main/DATASET.md) of K400, we got `82.1` test accuracy.
- Some other details can be found in [MMAction2 MViT page](https://github.com/open-mmlab/mmaction2/tree/dev-1.x/configs/recognition/mvit).

## Citation

```bibtex
@InProceedings{wei2022masked,
    author    = {Wei, Chen and Fan, Haoqi and Xie, Saining and Wu, Chao-Yuan and Yuille, Alan and Feichtenhofer, Christoph},
    title     = {Masked Feature Prediction for Self-Supervised Visual Pre-Training},
    booktitle = {CVPR},
    year      = {2022},
}
```

## Checklist

Here is a checklist illustrating a usual development workflow of a successful project, and also serves as an overview of this project's progress.

<!--The PIC (person in charge) or contributors of this project should check all the items that they believe have been finished, which will further be verified by codebase maintainers via a PR.

OpenMMLab's maintainer will review the code to ensure the project's quality. Reaching the first milestone means that this project suffices the minimum requirement of being merged into 'projects/'. But this project is only eligible to become a part of the core package upon attaining the last milestone.

Note that keeping this section up-to-date is crucial not only for this project's developers but the entire community, since there might be some other contributors joining this project and deciding their starting point from this list. It also helps maintainers accurately estimate time and effort on further code polishing, if needed.
A project does not necessarily have to be finished in a single PR, but it's essential for the project to at least reach the first milestone in its very first PR. -->

- [x] Milestone 1: PR-ready, and acceptable to be one of the `projects/`.

  - [x] Finish the code

    <!-- The code's design shall follow existing interfaces and convention. For example, each model component should be registered into `MMPretrain.registry.MODELS` and configurable via a config file. -->

  - [x] Basic docstrings & proper citation

    <!-- Each major object should contain a docstring, describing its functionality and arguments. If you have adapted the code from other open-source projects, don't forget to cite the source project in docstring and make sure your behavior is not against its license. Typically, we do not accept any code snippet under GPL license. [A Short Guide to Open Source Licenses](https://medium.com/nationwide-technology/a-short-guide-to-open-source-licenses-cf5b1c329edd) -->

  - [x] Inference correctness

    <!-- If you are reproducing the result from a paper, make sure your model's inference-time feature vectors or losses matches that from the original codes. The weights usually could be obtained by simply renaming the keys in the official pre-trained weights. This test could be skipped though, if you are able to prove the training-time correctness and check the second milestone. -->

  - [x] A full README

    <!-- As this template does. -->

- [x] Milestone 2: Indicates a successful model implementation.

  - [x] Training-time correctness

    <!-- If you are reproducing the result from a paper, checking this item means that you should have trained your model from scratch based on the original paper's specification and verified that the final result. Due to the pretrain-downstream pipeline of self-supervised learning, this item requires at least one downstream result matches the report within a minor error range. -->

- [ ] Milestone 3: Good to be a part of our core package!

  - [ ] Type hints and docstrings

    <!-- Ideally *all* the methods should have [type hints](https://www.pythontutorial.net/python-basics/python-type-hints/) and [docstrings](https://google.github.io/styleguide/pyguide.html#381-docstrings). [Example](https://github.com/open-mmlab/mmpretrain/blob/main/mmpretrain/models/selfsup/mae.py) -->

  - [ ] Unit tests

    <!-- Unit tests for each module are required. [Example](https://github.com/open-mmlab/mmpretrain/blob/main/tests/test_models/test_selfsup/test_mae.py) -->

  - [ ] Code polishing

    <!-- Refactor your code according to reviewer's comment. -->

  - [ ] `metafile.yml` and `README.md`

    <!-- It will be parsed by MIM and Inferencer. [Example](https://github.com/open-mmlab/mmpretrain/blob/main/configs/mae/metafile.yml). In particular, you may have to refactor this README into a standard one. [Example](https://github.com/open-mmlab/mmpretrain/blob/main/configs/mae/README.md) -->

- [ ] Refactor and Move your modules into the core package following the codebase's file hierarchy structure.
