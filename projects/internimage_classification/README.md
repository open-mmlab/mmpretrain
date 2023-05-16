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
mim train mmpretrain configs/internimage_t_1k_224.py \
	--work-dir work_dirs/internimage_t_1k_224/
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

Then, convert weights to mmpretrain. For example,

```bash
python tools/internimage_to_mmpretrain.py /PATH/TO/internimage_t_1k_224.pth tiny.pth
```

##### On Local Single GPU

```bash
# test with mim
mim test mmpretrain ${CONFIG} -C ${CHECKPOINT}

# a specific command example
mim test mmpretrain configs/internimage_t_1k_224.py -C tiny.pth
```

##### On Multiple GPUs

```bash
# test with mim
# a specific command examples, 8 GPUs here
mim test mmpretrain configs/internimage_t_1k_224.py \
	-C tiny.pth \
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

#### Fine-tune

With the converted weights, you can easily fine-tune models for downstream classification task like the below configuration,

```python
model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='InternImage',
        stem_channels=64,
        drop_path_rate=0.1,
        stage_blocks=[4, 4, 18, 4],
        groups=[4, 8, 16, 32],
        init_cfg=dict(
            type='Pretrained', checkpoint='/PATH/TO/internimage_tiny.pth', prefix='backbone')
        ),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='LinearClsHead',
        num_classes=NUM_CLASSES,
        in_channels=768,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        topk=(1, 5)))
```

## Results on ImageNet1K

The accuracy of different models on ImageNet1K,

|      name      |   pretrain   | resolution |  acc@1  |  acc@5  |                    config                    |
| :------------: | :----------: | :--------: | :-----: | :-----: | :------------------------------------------: |
| InternImage-T  | ImageNet-1K  |    224     | 83.3780 | 96.4100 | [config](./configs/internimage_t_1k_224.py)  |
| InternImage-S  | ImageNet-1K  |    224     | 83.9060 | 96.9300 | [config](./configs/internimage_s_1k_224.py)  |
| InternImage-B  | ImageNet-1K  |    224     | 84.5580 | 97.0700 | [config](./configs/internimage_b_1k_224.py)  |
| InternImage-L  | ImageNet-22K |    384     | 87.3760 | 98.2560 | [config](./configs/internimage_l_1k_384.py)  |
| InternImage-XL | ImageNet-22K |    384     | 87.6780 | 98.3880 | [config](./configs/internimage_xl_1k_384.py) |
| InternImage-H  |  Joint 427M  |    640     | 89.5500 | 98.8500 | [config](./configs/internimage_h_1k_640.py)  |
| InternImage-G  |      -       |    512     | 89.7520 | 98.9120 | [config](./configs/internimage_g_1k_512.py)  |

## Citation

```bibtex
@article{wang2022internimage,
  title={InternImage: Exploring Large-Scale Vision Foundation Models with Deformable Convolutions},
  author={Wang, Wenhai and Dai, Jifeng and Chen, Zhe and Huang, Zhenhang and Li, Zhiqi and Zhu, Xizhou and Hu, Xiaowei and Lu, Tong and Lu, Lewei and Li, Hongsheng and others},
  journal={arXiv preprint arXiv:2211.05778},
  year={2022}
}
```
