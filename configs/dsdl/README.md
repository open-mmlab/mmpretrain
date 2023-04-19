# DSDL: Standard Description Language for DataSet

## 1. Abstract

Data is the cornerstone of artificial intelligence. The efficiency of data acquisition, exchange, and application directly impacts the advances in technologies and applications. Over the long history of AI, a vast quantity of data sets have been developed and distributed. However, these datasets are defined in very different forms, which incurs significant overhead when it comes to exchange, integration, and utilization -- it is often the case that one needs to develop a new customized tool or script in order to incorporate a new dataset into a workflow.

To overcome such difficulties, we develop **Data Set Description Language (DSDL)**. More details please visit our [official documents](https://opendatalab.github.io/dsdl-docs/getting_started/overview/), dsdl datasets can be downloaded from our platform [OpenDataLab](https://opendatalab.com/).

## Steps

- install dsdl and opendatalab:

  ```
  pip install dsdl
  pip install opendatalab
  ```

- install mmpretrain and pytorch:
  please refer this [installation documents](https://mmpretrain.readthedocs.io/en/latest/get_started.html).

- prepare dsdl dataset (take cifar10 as an example)

  - download dsdl dataset (you will need an opendatalab account to do so. [register one now](https://opendatalab.com/))

    ```
    cd data

    odl login
    odl get CIFAR-10
    ```

    usually, dataset is compressed on opendatalab platform, the downloaded cifar10 dataset should be like this:

    ```
    data/
    ├── CIFAR-10
    │   ├── dsdl
    │   │   └── dsdl_Cls_full.zip
    │   ├── raw
    │   │   ├── cifar-10-binary.tar.gz
    │   │   ├── cifar-10-matlab.tar.gz
    │   │   └── cifar-10-python.tar.gz
    │   └── README.md
    └── ...
    ```

  - decompress dataset

    decompress dsdl files:

    ```
    cd dsdl
    unzip dsdl_Cls_full.zip
    ```

    decompress raw data and save as image files, we prepared a python script to do so:

    ```
    cd ..
    python dsdl/dsdl_Cls_full/tools/prepare.py raw/

    cd ../../
    ```

    after running this script, there will be a new folder named as `prepared` (this does not happen on every dataset, for cifar10 has binary files and needs to be extracted as image files):

    ```
    data/
    ├── CIFAR-10
    │   ├── dsdl
    │   │   └── ...
    │   ├── raw
    │   │   └── ...
    │   ├── prepared
    │   │   └── images
    │   └── README.md
    └── ...
    ```

- change traning config

  open the [cifar10 config file](cifar10.py) and set some file paths as below:

  ```
  data_root = 'data/CIFAR-10'
  img_prefix = 'prepared'
  train_ann = 'dsdl/dsdl_Cls_full/set-train/train.yaml'
  val_ann = 'dsdl/dsdl_Cls_full/set-test/test.yaml'
  ```

  as dsdl datasets with one task using one dataloader, we can simplly change these file paths to train a model on a different dataset.

- train:

  - using single gpu:

  ```
  python tools/train.py {config_file}
  ```

  - using slrum:

  ```
  ./tools/slurm_train.sh {partition} {job_name} {config_file} {work_dir} {gpu_nums}
  ```

## 3. Test Results

|  Datasets   |                                                      Model                                                      | Top-1 Acc (%) |          Config           |
| :---------: | :-------------------------------------------------------------------------------------------------------------: | :-----------: | :-----------------------: |
|   cifar10   | [model](https://download.openmmlab.com/mmclassification/v0/resnet/resnet18_b16x8_cifar10_20210528-bd6371c8.pth) |     94.83     |  [config](./cifar10.py)   |
| ImageNet-1k |  [model](https://download.openmmlab.com/mmclassification/v0/resnet/resnet18_8xb32_in1k_20210831-fbbb1da6.pth)   |     69.84     | [config](./imagenet1k.py) |
