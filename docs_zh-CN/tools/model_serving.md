# 模型部署至 TorchServe

为了使用 [`TorchServe`](https://pytorch.org/serve/) 部署一个 `MMClassification` 模型，需要进行以下几步：

## 1. 转换 MMClassification 模型至 TorchServe

```shell
python tools/deployment/mmcls2torchserve.py ${CONFIG_FILE} ${CHECKPOINT_FILE} \
--output-folder ${MODEL_STORE} \
--model-name ${MODEL_NAME}
```

**注意**: ${MODEL_STORE} 需要是一个文件夹的绝对路径。

## 2. 构建 `mmcls-serve` docker 镜像

```shell
docker build -t mmcls-serve:latest docker/serve/
```

## 3. 运行 `mmcls-serve` 镜像

请参考官方文档 [基于 docker 运行 TorchServe](https://github.com/pytorch/serve/blob/master/docker/README.md#running-torchserve-in-a-production-docker-environment).

为了使镜像能够使用 GPU 资源，需要安装 [nvidia-docker](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)。之后可以传递 `--gpus` 参数以在 GPU 上运。

示例：

```shell
docker run --rm \
--cpus 8 \
--gpus device=0 \
-p8080:8080 -p8081:8081 -p8082:8082 \
--mount type=bind,source=$MODEL_STORE,target=/home/model-server/model-store \
mmcls-serve:latest
```

参考 [该文档](https://github.com/pytorch/serve/blob/master/docs/rest_api.md) 了解关于推理 (8080)，管理 (8081) 和指标 (8082) 等 API 的信息。

## 4. 测试部署

```shell
curl -O https://raw.githubusercontent.com/pytorch/serve/master/docs/images/3dogs.jpg
curl http://127.0.0.1:8080/predictions/${MODEL_NAME} -T 3dogs.jpg
```

您应该获得类似于以下内容的响应：

```json
{
  "pred_label": 245,
  "pred_score": 0.5536593794822693,
  "pred_class": "French bulldog"
}
```
