# Model Serving

In order to serve an `MMClassification` model with [`TorchServe`](https://pytorch.org/serve/), you can follow the steps:

## 1. Convert model from MMClassification to TorchServe

```shell
python tools/deployment/mmcls2torchserve.py ${CONFIG_FILE} ${CHECKPOINT_FILE} \
--output-folder ${MODEL_STORE} \
--model-name ${MODEL_NAME}
```

**Note**: ${MODEL_STORE} needs to be an absolute path to a folder.

## 2. Build `mmcls-serve` docker image

```shell
docker build -t mmcls-serve:latest docker/serve/
```

## 3. Run `mmcls-serve`

Check the official docs for [running TorchServe with docker](https://github.com/pytorch/serve/blob/master/docker/README.md#running-torchserve-in-a-production-docker-environment).

In order to run in GPU, you need to install [nvidia-docker](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html). You can omit the `--gpus` argument in order to run in GPU.

Example:

```shell
docker run --rm \
--cpus 8 \
--gpus device=0 \
-p8080:8080 -p8081:8081 -p8082:8082 \
--mount type=bind,source=$MODEL_STORE,target=/home/model-server/model-store \
mmcls-serve:latest
```

[Read the docs](https://github.com/pytorch/serve/blob/master/docs/rest_api.md) about the Inference (8080), Management (8081) and Metrics (8082) APis

## 4. Test deployment

```shell
curl -O https://raw.githubusercontent.com/pytorch/serve/master/docs/images/3dogs.jpg
curl http://127.0.0.1:8080/predictions/${MODEL_NAME} -T 3dogs.jpg
```

You should obtain a respose similar to:

```json
{
  "pred_label": 245,
  "pred_score": 0.5536593794822693,
  "pred_class": "French bulldog"
}
```
