# InternImage Classification

## Description

This is the implementation of [InternImage](https://arxiv.org/abs/2211.05778) for image classification.

## Usage

### Setup Environment

Please refer to [Get Started](https://mmpretrain.readthedocs.io/en/latest/get_started.html) documentation of MMPretrain to finish installation.

Please install DCNv3. Run the command below following the [ InternImage official installation instructions](https://github.com/OpenGVLab/InternImage/blob/master/classification/README.md).

```shell
cd ops_dcnv3
sh ./make.sh
```

### Training and Test Commands

At first, you need to add the current folder to `PYTHONPATH`, so that Python can find your model files. In `projects/internimage_classification/` root directory, please run command below to add it.

```shell
export PYTHONPATH=`pwd`:$PYTHONPATH
```

#### Training

##### On Local Single GPU

```bash
# train with mim
mim train mmpretrain ${CONFIG} --work-dir ${WORK_DIR}

# a specific command example
mim train mmpretrain configs/internimage-tiny_8xb128_in1k-224.py \
	--work-dir work_dirs/internimage-tiny_8xb128_in1k-224/
```

##### On Multiple GPUs

```bash
# train with mim
mim train mmpretrain ${CONFIG} \
    --work-dir ${WORK_DIR} \
    --launcher pytorch --gpus 8
```

##### On Multiple GPUs with Slurm

```bash
# train with mim
mim train mmpretrain ${CONFIG} \
    --work-dir ${WORK_DIR} \
    --launcher slurm --gpus 16 --gpus-per-node 8 \
    --partition ${PARTITION}
```

#### Test

Please download the pretrain weight provided by [OpenGVLab](https://github.com/OpenGVLab/) from [here](https://huggingface.co/OpenGVLab/InternImage/tree/main)

##### On Local Single GPU

```bash
# test with mim
mim test mmpretrain ${CONFIG} -C ${CHECKPOINT}

# a specific command example
mim test mmpretrain configs/internimage-tiny_8xb128_in1k-224.py -C /PATH/TO/internimage_t_1k_224.pth
```

##### On Multiple GPUs

```bash
# test with mim
# a specific command examples, 8 GPUs here
mim test mmpretrain configs/internimage_t_1k_224.py \
	-C /PATH/TO/internimage_t_1k_224.pth \
    --launcher pytorch --gpus 8
```

##### On Multiple GPUs with Slurm

```bash
# test with mim
mim test mmpretrain ${CONFIG} \
    -C ${CHECKPOINT}
    --work-dir ${WORK_DIR} \
    --launcher slurm --gpus 8 --gpus-per-node 8 \
    --partition ${PARTITION} \
    $PY_ARGS
```

Note: `PY_ARGS` is other optional args.

## Results on ImageNet1K

The accuracy of different models on ImageNet1K,

|      name      | resolution |  acc@1  |  acc@5  |                          config                           |                                              weight                                               |
| :------------: | :--------: | :-----: | :-----: | :-------------------------------------------------------: | :-----------------------------------------------------------------------------------------------: |
| InternImage-T  |    224     | 83.4700 | 96.5340 |  [config](./configs/internimage-tiny_8xb128_in1k-224.py)  |    [model](https://huggingface.co/OpenGVLab/InternImage/resolve/main/internimage_t_1k_224.pth)    |
| InternImage-S  |    224     | 84.1640 | 96.9320 | [config](./configs/internimage-small_8xb128_in1k-224.py)  |    [model](https://huggingface.co/OpenGVLab/InternImage/resolve/main/internimage_s_1k_224.pth)    |
| InternImage-B  |    224     | 84.8660 | 97.1820 |  [config](./configs/internimage-base_8xb128_in1k-224.py)  |    [model](https://huggingface.co/OpenGVLab/InternImage/resolve/main/internimage_b_1k_224.pth)    |
| InternImage-L  |    384     | 87.7060 | 98.3820 | [config](./configs/internimage-large_8xb128_in1k-384.py)  | [model](https://huggingface.co/OpenGVLab/InternImage/resolve/main/internimage_l_22kto1k_384.pth)  |
| InternImage-XL |    384     | 88.0460 | 98.5620 | [config](./configs/internimage-xlagre_8xb128_in1k-384.py) | [model](https://huggingface.co/OpenGVLab/InternImage/resolve/main/internimage_xl_22kto1k_384.pth) |
| InternImage-H  |    640     | 89.5500 | 98.8500 |  [config](./configs/internimage-huge_8xb128_in1k-640.py)  | [model](https://huggingface.co/OpenGVLab/InternImage/resolve/main/internimage_h_22kto1k_640.pth)  |
| InternImage-G  |    512     | 90.0580 | 98.9700 | [config](./configs/internimage-giant_8xb128_in1k-512.py)  | [model](https://huggingface.co/OpenGVLab/InternImage/resolve/main/internimage_g_22kto1k_512.pth)  |

## Citation

```bibtex
@article{wang2022internimage,
  title={InternImage: Exploring Large-Scale Vision Foundation Models with Deformable Convolutions},
  author={Wang, Wenhai and Dai, Jifeng and Chen, Zhe and Huang, Zhenhang and Li, Zhiqi and Zhu, Xizhou and Hu, Xiaowei and Lu, Tong and Lu, Lewei and Li, Hongsheng and others},
  journal={arXiv preprint arXiv:2211.05778},
  year={2022}
}
```
