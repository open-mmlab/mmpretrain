# 推理命令

一共十七个命令

## ViT

```shell
PORT=8888 GPUS=1 BATCH_SIZE=32 bash tools/run_single_inference.sh configs/vit/l-448-arc-rounda3-0.4.py checkpoints/l-448-arc-rounda3-0.4.pth pkls/l-448-arc-rounda3-0.4-7473.pkl
```

```shell
PORT=8888 GPUS=1 BATCH_SIZE=32 bash tools/run_single_inference.sh configs/vit/l-448-arc-rounda3-0.5.py checkpoints/l-448-arc-rounda3-0.5.pth pkls/l-448-arc-rounda3-0.5-7418.pkl
```

```shell
PORT=8888 GPUS=1 BATCH_SIZE=32 bash tools/run_single_inference.sh configs/vit/l-448-arc-roundb1.py checkpoints/l-448-arc-roundb1.pth pkls/l-448-arc-roundb1-7520.pkl
```

```shell
PORT=8888 GPUS=1 BATCH_SIZE=32 bash tools/run_single_inference.sh configs/vit/l-448-arc-roundb2.py checkpoints/l-448-arc-roundb2.pth pkls/l-448-arc-roundb2-7570.pkl
```

```shell
PORT=8888 GPUS=1 BATCH_SIZE=32 bash tools/run_single_inference.sh configs/vit/l-448-arc-roundb3.py checkpoints/l-448-arc-roundb3.pth pkls/l-448-arc-roundb3-7620.pkl
```

```shell
PORT=8888 GPUS=1 BATCH_SIZE=32 bash tools/run_single_inference.sh configs/vit/l-448-ce-rounda3-0.4.py checkpoints/l-448-ce-rounda3-0.4.pth pkls/l-448-ce-rounda3-0.4-7471.pkl --lt
```

```shell
PORT=8888 GPUS=1 BATCH_SIZE=32 bash tools/run_single_inference.sh configs/vit/l-448-ce-rounda3-0.5.py checkpoints/l-448-ce-rounda3-0.5.pth pkls/l-448-ce-rounda3-0.5-7405.pkl --lt
```

```shell
PORT=8888 GPUS=1 BATCH_SIZE=32 bash tools/run_single_inference.sh configs/vit/l-448-ce-roundb1.py checkpoints/l-448-ce-roundb1.pth pkls/l-448-ce-roundb1-7450.pkl --lt
```

```shell
PORT=8888 GPUS=1 BATCH_SIZE=32 bash tools/run_single_inference.sh configs/vit/l-448-ce-roundb2.py checkpoints/l-448-ce-roundb2.pth pkls/l-448-ce-roundb2-7500.pkl --lt
```

```shell
PORT=8888 GPUS=1 BATCH_SIZE=32 bash tools/run_single_inference.sh configs/vit/l-448-ce-roundb3.py checkpoints/l-448-ce-roundb3.pth pkls/l-448-ce-roundb3-7550.pkl --lt
```

## Swin

```shell
PORT=8888 GPUS=1 BATCH_SIZE=32 bash tools/run_single_inference.sh configs/swin/b-384-arc-roundb1.py checkpoints/swin-b-384-arc-roundb1-7410.pth pkls/swin-b-384-arc-roundb1-7410.pkl
```

```shell
PORT=8888 GPUS=1 BATCH_SIZE=32 bash tools/run_single_inference.sh configs/swin/b-384-arc-roundb2.py checkpoints/swin-b-384-arc-roundb2-7460.pth pkls/swin-b-384-arc-roundb2-7460.pkl
```

```shell
PORT=8888 GPUS=1 BATCH_SIZE=32 bash tools/run_single_inference.sh configs/swin/b-384-arc-roundb3.py checkpoints/swin-b-384-arc-roundb3-7510.pth pkls/swin-b-384-arc-roundb3-7510.pkl
```

```shell
PORT=8888 GPUS=1 BATCH_SIZE=32 bash tools/run_single_inference.sh configs/swin/l-384-arc-round2.py checkpoints/swin-l-384-arc-round2-7510.pth pkls/swin-l-384-arc-round2-7510.pkl
```

```shell
PORT=8888 GPUS=1 BATCH_SIZE=32 bash tools/run_single_inference.sh configs/swin/l-384-arc-rounda3.py checkpoints/swin-l-384-arc-rounda3-7400.pth pkls/swin-l-384-arc-rounda3-7400.pkl
```

```shell
PORT=8888 GPUS=1 BATCH_SIZE=32 bash tools/run_single_inference.sh configs/swin/l-384-arc-roundb1.py checkpoints/swin-l-384-arc-roundb1-7460.pth pkls/swin-l-384-arc-roundb1-7460.pkl
```

```shell
PORT=8888 GPUS=1 BATCH_SIZE=32 bash tools/run_single_inference.sh configs/swin/l-384-arc-roundb3.py checkpoints/swin-l-384-arc-roundb3-7560.pth pkls/swin-l-384-arc-roundb3-7560.pkl
```

**Note:** PORT, GPUS, BATCH_SIZE 分别为分布式通讯端口, GPUS 为显卡数量, BATCH_SIZE 为推理的 batch size。这几个参数需要根据实际情况设定。
