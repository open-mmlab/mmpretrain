# Fine-tuning MAE on Some Image Classification Datasets

## Usage

### Setup Environment

Please refer to [Get Started](https://mmpretrain.readthedocs.io/en/latest/get_started.html) documentation of MMPretrain to finish installation.

### Data Preparation

Please download and unzip datasets in the `data` folder.

### Fine-tuning Commands

At first, you need to add the current folder to `PYTHONPATH`, so that Python can find your model files. In `projects/mae_classification/` root directory, please run command below to add it.

```shell
export PYTHONPATH=`pwd`:$PYTHONPATH
```

Then run the following commands to train the model:

#### On Local Single GPU

```bash
# train with mim
mim train mmpretrain ${CONFIG} --work-dir ${WORK_DIR}

# a specific command example
mim train mmpretrain configs/vit-base-p16_8xb8-coslr-100e_caltech101.py --work-dir work_dirs/vit-base-p16_8xb8-coslr-100e_caltech101
```

#### On Multiple GPUs

```bash
# train with mim
# a specific command examples, 8 GPUs here
mim train mmpretrain configs/vit-base-p16_8xb8-coslr-100e_caltech101.py --work-dir work_dirs/vit-base-p16_8xb8-coslr-100e_caltech101 --launcher pytorch --gpus 8
```

Note:

- CONFIG: the config files under the directory `configs/`
- WORK_DIR: the working directory to save configs, logs, and checkpoints

#### On Multiple GPUs with Slurm

```bash
# train with mim
mim train mmpretrain ${CONFIG} \
    --work-dir ${WORK_DIR} \
    --launcher slurm --gpus 16 --gpus-per-node 8 \
    --partition ${PARTITION}
```

Note:

- CONFIG: the config files under the directory `configs/`
- WORK_DIR: the working directory to save configs, logs, and checkpoints
- PARTITION: the slurm partition you are using

## Results

|      Datasets       | Backbone |  Params  |    Flops    | Accuracy (%) |                             Config                              |
| :-----------------: | :------: | :------: | :---------: | :----------: | :-------------------------------------------------------------: |
|      Food-101       | MAE-base | 85876325 | 17581219584 |    91.57     |   [config](configs/vit-base-p16_8xb32-coslr-100e_food101.py)    |
|      CIFAR-10       | MAE-base | 85806346 | 17581219584 |    98.45     |   [config](configs/vit-base-p16_8xb32-coslr-100e_cifar10.py)    |
|      CIFAR-100      | MAE-base | 85875556 | 17581219584 |    90.06     |   [config](configs/vit-base-p16_8xb16-coslr-100e_cifar100.py)   |
|       SUN397        | MAE-base | 86103949 | 17581219584 |    67.84     |    [config](configs/vit-base-p16_8xb32-coslr-100e_sun397.py)    |
|    Stanford Cars    | MAE-base | 85949380 | 17581219584 |    93.11     | [config](configs/vit-base-p16_8xb8-coslr-100e_stanfordcars.py)  |
|    FGVC Aircraft    | MAE-base | 85875556 | 17581219584 |    88.24     | [config](configs/vit-base-p16_8xb8-coslr-100e_fgvcaircraft.py)  |
|         DTD         | MAE-base | 85834799 | 17581219584 |    77.55     |     [config](configs/vit-base-p16_8xb16-coslr-100e_dtd.py)      |
|  Oxford-IIIT Pets   | MAE-base | 85827109 | 17581219584 |    91.66     | [config](configs/vit-base-p16_8xb8-coslr-100e_oxfordiiitpet.py) |
|     Caltech-101     | MAE-base | 85877094 | 17581219584 |    93.22     |  [config](configs/vit-base-p16_8xb8-coslr-100e_caltech101.py)   |
| Oxford 102 Flowers  | MAE-base | 85877094 | 17581219584 |    95.20     |  [config](configs/vit-base-p16_8xb8-coslr-100e_flowers102.py)   |
| PASCAL VOC 2007 cls | MAE-base | 85814036 | 17581219584 | 88.69 (mAP)  |      [config](configs/vit-base-p16_8xb8-coslr-100e_voc.py)      |

## Citation

```bibtex
@article{He2021MaskedAA,
  title={Masked Autoencoders Are Scalable Vision Learners},
  author={Kaiming He and Xinlei Chen and Saining Xie and Yanghao Li and
  Piotr Doll'ar and Ross B. Girshick},
  journal={arXiv},
  year={2021}
}
```
