# DSDL: Standard Description Language for DataSet

## 1. Abstract

Data is the cornerstone of artificial intelligence. The efficiency of data acquisition, exchange, and application directly impacts the advances in technologies and applications. Over the long history of AI, a vast quantity of data sets have been developed and distributed. However, these datasets are defined in very different forms, which incurs significant overhead when it comes to exchange, integration, and utilization -- it is often the case that one needs to develop a new customized tool or script in order to incorporate a new dataset into a workflow.

To overcome such difficulties, we develop **Data Set Description Language (DSDL)**. More details please visit our [official documents](https://opendatalab.github.io/dsdl-docs/getting_started/overview/), dsdl datasets can be downloaded from our platform [OpenDataLab](https://opendatalab.com/).

## 2. Steps

- install dsdl:

  install by pip:

  ```
  pip install dsdl
  ```

  install by source code:

  ```
  git clone https://github.com/opendatalab/dsdl-sdk.git -b schema-dsdl
  cd dsdl-sdk
  python setup.py install
  ```

- install mmdet and pytorch:
  please refer this [installation documents](https://mmdetection.readthedocs.io/en/3.x/get_started.html).

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
