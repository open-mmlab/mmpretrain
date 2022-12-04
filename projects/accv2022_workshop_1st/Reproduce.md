# 复现精度

## 准备

### 解压代码以及下载资源

百度网盘

链接: https://pan.baidu.com/s/1SVmwQaVkrUvbySJGYb9q4A 提取码: 2yza

应该会在11/16日凌晨2-3点传完

- checkpoints   17个模型权重  '.pth'
- testb-pkls    17个testb的推理结果 '.pkl'

##### 下载pkls

pkl的目录:

```shell
testb-pkls (17个pkl)
    ├── mae-vit-ce-30e_semi-3st-thres4-7471.pkl       # 7410 为精度，必须加上，
    ├── mae-vit-ce-30e-testb-3st-7550.pkl             # rounda为LBa提交精度，roundb为估计
    ├── swin-l-384-arc-rounda3-testb-7400.pkl
    ├── vit-448-arc-30e-3st-thr4-semi-7473.pkl
    ├── vit-448-arc-30e-testb-3st-7620.pkl
    ├── mae-vit-ce-30e_semi-3st-thres5-7405.pkl
    ├── swin-b-384-arc-roundb1-testb-7410.pkl
    ├── swin-l-384-arc-roundb1-testb-7460.pkl
    ├── vit-448-arc-30e-3st-thr5-semi-7418.pkl
    ├── mae-vit-ce-30e-testb-1st-7450.pkl
    ├── swin-b-384-arc-roundb2-testb-7460.pkl
    ├── swin-l-384-arc-roundb2-testb-7510.pkl
    ├── vit-448-arc-30e-testb-1st-7520.pkl
    ├── mae-vit-ce-30e-testb-2st-7500.pkl
    ├── swin-b-384-arc-roundb3-testb-7510.pkl
    ├── swin-l-384-arc-roundb3-testb-7560.pkl
    ├── vit-448-arc-30e-testb-2st-7570.pkl
```

##### 下载checkpoints

项目结构：

```
$ACCV_workshop
├── config                         # 所有的 config
│   ├── swin                       # 所有的 swin 相关config
│   │   ├── _base_                 # 基础配置
│   │   ├── b-384-arc_roundb1.py
│   │   └── ......
│   └── vit                         # 所有的 vit 相关config
|        ├── _base_                 # 基础配置
|        ├── b-384-arc_roundb1.py
|        └── ......
├── data                             # 数据集
├── checkpoints                      # 存放所有的权重
├── pretrain-checkpoints             # 存放预训练权重
├── testb-pkls                       # 存放所有的推理间接结果
├── src                              # 所有源代码
├── tools                            # 所有训练测试以及各种工具脚本
├── docker/Dockerfile                # docker镜像
├── docker-compose.yml
├── submit/pred_result.csv            # 生成的.csv文件
├── submit/pred_result.zip            # 用于提交的.zip文件 此处为提交的最好结果
├── requirements.txt
└── README.md
```

### 准备数据集

1. 下载所有的数据集并解压为 train, testa, testb 文件夹，放入`data/ACCV_workshop/` 文件夹下并进入；

2. 在 train 文件夹下建立指向 testa, testb 的**软链接**

   ```shell
   cd data/ACCV_workshop/
   ln -s ./testa train/testa
   ln -s ./testb train/testb
   ```

3. 下载所有的 meta 文件并解压,如果存在，不需要处理; (也可以从网盘下载： 链接: https://pan.baidu.com/s/1SVmwQaVkrUvbySJGYb9q4A 提取码: 2yza)

   ```shell
   # wget -O  "meta.zip" https://tmp-titan.vx-cdn.com/file-6373827f4b03a-637396faadc56/meta.zip
   wget -O  "meta.zip" https://tmp-titan.vx-cdn.com/file-637f01a37561f-637f023c53309/meta.zip
   unzip meta.zip
   ```

最后数据集的目录结构为:

```shell
data/ACCV_workshop
    ├── train
    │   ├── testa   # 指向 testa 的**软链接**, soft link
    │   ├── testb   # 指向 testb 的**软链接**, soft link
    │   ├── 0000
    │   ├── 0001
    │   └── ......
    ├── testa
    │   ├── xxxxxx.jpg
    │   ├── yyyyyy.jpg
    │   └── ......
    ├── testb
    │   ├── 1111111.jpg
    │   ├── 222222.jpg
    │   └── ......
    ├── meta
    │   ├── train.txt              # 数据清洗后的训练集标注
    │   ├── all.txt                # 数据清洗前的训练集标注
    │   ├── rounda1/train.txt      # 加入testa的pseudo，第1轮训练集标注
    │   ├── rounda2/train.txt      # 加入testa的pseudo，第2轮训练集标注
    │   ├── rounda3/train.txt      # 加入testa的pseudo，第3轮训练集标注
    │   ├── roundb1/train.txt      # 加入testa以及testb的pseudo，第1轮训练集标注
    │   ├── roundb2/train.txt      # 加入testa以及testb的pseudo，第2轮训练集标注
    │   └── roundb3/train.txt      # 加入testa以及testb的pseudo，第3轮训练集标注
```

### 启动

在 ACCV_workshop 项目根目录下。

构建镜像：

```
docker build -t openmmlab:accv docker/
```

启动容器：

```
docker run -it \
    -v $PWD:/workspace/ACCV_workshop \
    -w /workspace/ACCV_workshop \
    --gpus all \
    --shm-size 128g \
    -e PYTHONPATH=/working:$PYTHONPATH \
    openmmlab:accv  /bin/bash
```

也可以使用 docker-compose， (如果希望使用，则后面的命令需要根据docker-compose修改)

```shell
docker-compose up -d accv
```

## 集成以及re-distribute-label (快速得到结果)

pkl的目录:

```shell
testb-pkls (17个pkl)
    ├── swin-b-384-arc-roundb1-testb-7410.pkl         # 7410 为精度，必须加上，
    ├── swin-b-384-arc-roundb2-testb-7460.pkl         # rounda为LBa提交精度，roundb为估计
    ├── mae-vit-ce-30e_semi-3st-thres4-7471.pkl
    ├── ....
    └── vit-448-arc-30e-testb-3st-7620.pkl
```

**注意**： 每个 pkl 文件对应每个模型在 testb 数据集上的推理结果，其每条数据为一个 5000
维度的向量, 对应该样本属于 5000 类中每一类的概率。

#### 集成

```
python tools/emsemble.py --pkls-dir testb-pkls --factor 25 --scale --dump-all testb.pkl
```

看到：

```text
Number of .pkls is 17....
Adjusted factor is: [2.080085996252569, 1.3987451430721878, 1.2237077021500773, 1.0626193781723274, 1.1833516629706529, 1.0, 1.764428216037884, 1.0343373472253807, 1.446123660572264, 1.7070722173927184, 1.2781492311812328, 1.269624858196828, 1.017029565305733, 1.6515084398827093, 1.2237077021500773, 1.446123660572264, 1.495040719791741]
.......
```

可以 `zip pred_results.zip pred_results.csv` 打包提交得到 **0.7815**, 'testb.pkl' 是临时中间结果，给下面使用。

#### re-distribute-label (调整预测的分布)

```
python tools/re-distribute-label.py testb.pkl --K 16
```

可以 `zip pred_results.zip pred_results.csv` 打包提交得到 **0.7894**,

## 推理

上面的 pkl 文件为模型在 testb 上的推理结果，这个文件 [CMD](./tools/CMD.md) 包含了我们推理的所有命令, 但是首先需要使用以下命令在根目录下创建一个文件夹用于保存推理生成的文件:

```shell
mkdir pkls
```

单个 pkl，生成 pred_result.csv 以及生成 pred_result.zip

```
python tools/emsemble.py --pkls pkls/XXXXXXXX.pkl
zip pred_results.zip pred_results.csv
```

多个 pkl，生成 pred_result.csv 以及生成 pred_result.zip

```
python tools/emsemble.py --pkls-dir pkls
zip pred_results.zip pred_results.csv
```

## Fine-tuning

以上用于推理的模型都是由 fine-tuning 得到，因为我们所有的模型均为 SwinTransformer 和 ViT, 下面我们分别给出这两者的 fine-tuning 命令:

### ViT

**模型**: ViT-L
**训练耗时**: 16 卡 80G A100, 预估 20h;

**预训练**: ViT-L 的预训练为 ViT-L 在官方提供的 train+testa 上使用 MAE 预训练 1600e,
预训练采用了[MMSelfSup](https://github.com/open-mmlab/mmselfsup/tree/1.x), 配置文件为
[config](https://github.com/open-mmlab/mmselfsup/blob/1.x/configs/selfsup/mae/mae_vit-large-p16_8xb512-amp-coslr-1600e_in1k.py), 预训练初始化的权重为 [pretrain_init](https://download.openmmlab.com/mmselfsup/1.x/mae/mae_vit-large-p16_8xb512-fp16-coslr-1600e_in1k/mae_vit-large-p16_8xb512-fp16-coslr-1600e_in1k_20220825-cc7e98c9.pth), 预训练得到的权重为 [weight](./pretrain-checkpoints/epoch_1600.pth). PS: 预训练采用 16 卡 80G A100，训练时长大概为一周。

**PyTorch**

```
GPUS=16 SRUN_ARGS="--preempt --quotatype=reserved --async" bash tools/mim_slurm_train.sh mm_model configs/vit/l-448-arc-rounda3-0.4.py pretrain-checkpoints/epoch_1600.pth "env_cfg.cudnn_benchmark=True train_dataloader.batch_size=16 train_dataloader.num_workers=16 train_dataloader.persistent_workers=True default_hooks.checkpoint.max_keep_ckpts=1 resume=False"
```

**Slurm**

```
GPUS_PER_NODE=8 GPUS=16 CPUS_PER_TASK=16 SRUN_ARGS="--preempt --quotatype=reserved --async" bash tools/mim_slurm_train.sh mm_model configs/vit/l-448-arc-rounda3-0.4.py pretrain-checkpoints/epoch_1600.pth "env_cfg.cudnn_benchmark=True train_dataloader.batch_size=16 train_dataloader.num_workers=16 train_dataloader.persistent_workers=True default_hooks.checkpoint.max_keep_ckpts=1 resume=False"
```

### Swin

**所需时间**
swin-b 需要 8张卡， swin-l 需要 16 张卡； 预估 15 h；

**预训练**
使用预训练，都是 21k 上的预训练， 来自[MMCls](https://github.com/open-mmlab/mmclassification/tree/dev-1.x/configs/swin_transformer_v2).

- [swin-b-21kpt](https://download.openmmlab.com/mmclassification/v0/swin-v2/pretrain/swinv2-base-w12_3rdparty_in21k-192px_20220803-f7dc9763.pth)
- [swin-l-21kpt](https://download.openmmlab.com/mmclassification/v0/swin-v2/pretrain/swinv2-large-w12_3rdparty_in21k-192px_20220803-d9073fee.pth)

**PyTorch**

```
python -m torch.distributed.launch --nproc_per_node=16  tools/train.py configs/swin/l-384-arc_roundb3.py ~/accv/l-384-arc_roundb3  --amp
```

**Slurm**

```
GPUS=16 sh ./tools/slurm_train.sh ${CLUSTER_NAME} ${JOB_NAME} configs/swin/l-384-arc_roundb3.py ~/accv/l-384-arc_roundb3 --amp
```

#### Uniform Model Soup

融合训练得到的模型， swin-b 融合最后 5个， swin-l 融合 最后的 7 个。将需要的checkpoint 放在一个文件夹中。使用以下命令

```
python tools/model_soup.py --model-folder ${CKPT-DIR} --out ${Final-CKPT}
```

## 伪标签

Get needed inter-result

```shell
python tools/emsemble.py --pkls-dir testb-pkls --dump testb-pseudo.pkl  # 不要加 --scale --factor 25
python tools/creat_pseudo.py testb-pseudo.pkl --thr 0.45 --testb
```

可以得到：

```text
90000 samples have been found....
Get 78458 pseudo samples....
```

注意：

1. 第一步不要'--scale --factor 25'
2. 第二部需要根据当前的数据集， `--testb`表示生成的 'testb' 的标签， 不加为 'testa'的标签；
   区别为生成的 'pseudo.txt' 中的图片路径前缀不同，分别为 'testb' 和 'testa'
3. 生成的 'pseudo.txt' 需要和之前的训练标注合并起来才能使用
4. 想使用必须在 './data/ACCV_workshop/train' 建立 'testa' 与 'testb' 的软连接。
5. roundax 使用训练集与testa伪标签训练，roundbx 使用训练集与testa与testb伪标签训练，
