# TorchServe 部署

为了使用 [`TorchServe`](https://pytorch.org/serve/) 部署一个 `MMPretrain` 模型，需要进行以下几步：

## 1. 转换 MMPretrain 模型至 TorchServe

```shell
python tools/torchserve/mmpretrain2torchserve.py ${CONFIG_FILE} ${CHECKPOINT_FILE} \
--output-folder ${MODEL_STORE} \
--model-name ${MODEL_NAME}
```

```{note}
${MODEL_STORE} 需要是一个文件夹的绝对路径。
```

示例：

```shell
python tools/torchserve/mmpretrain2torchserve.py \
  configs/resnet/resnet18_8xb32_in1k.py \
  checkpoints/resnet18_8xb32_in1k_20210831-fbbb1da6.pth \
  --output-folder ./checkpoints \
  --model-name resnet18_in1k
```

## 2. 构建 `mmpretrain-serve` docker 镜像

```shell
docker build -t mmpretrain-serve:latest docker/serve/
```

## 3. 运行 `mmpretrain-serve` 镜像

请参考官方文档 [基于 docker 运行 TorchServe](https://github.com/pytorch/serve/blob/master/docker/README.md#running-torchserve-in-a-production-docker-environment).

为了使镜像能够使用 GPU 资源，需要安装 [nvidia-docker](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)。之后可以传递 `--gpus` 参数以在 GPU 上运。

示例：

```shell
docker run --rm \
--name mar \
--cpus 8 \
--gpus device=0 \
-p8080:8080 -p8081:8081 -p8082:8082 \
--mount type=bind,source=`realpath ./checkpoints`,target=/home/model-server/model-store \
mmpretrain-serve:latest
```

```{note}
`realpath ./checkpoints` 是 "./checkpoints" 的绝对路径，你可以将其替换为你保存 TorchServe 模型的目录的绝对路径。
```

参考 [该文档](https://github.com/pytorch/serve/blob/master/docs/rest_api.md) 了解关于推理 (8080)，管理 (8081) 和指标 (8082) 等 API 的信息。

## 4. 测试部署

```shell
curl http://127.0.0.1:8080/predictions/${MODEL_NAME} -T demo/demo.JPEG
```

您应该获得类似于以下内容的响应：

```json
{
  "pred_label": 58,
  "pred_score": 0.38102269172668457,
  "pred_class": "water snake"
}
```

另外，你也可以使用 `test_torchserver.py` 来比较 TorchServe 和 PyTorch 的结果，并进行可视化。

```shell
python tools/torchserve/test_torchserver.py ${IMAGE_FILE} ${CONFIG_FILE} ${CHECKPOINT_FILE} ${MODEL_NAME}
[--inference-addr ${INFERENCE_ADDR}] [--device ${DEVICE}]
```

示例：

```shell
python tools/torchserve/test_torchserver.py \
  demo/demo.JPEG \
  configs/resnet/resnet18_8xb32_in1k.py \
  checkpoints/resnet18_8xb32_in1k_20210831-fbbb1da6.pth \
  resnet18_in1k
```
