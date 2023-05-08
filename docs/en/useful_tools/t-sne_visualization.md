# t-Distributed Stochastic Neighbor Embedding (t-SNE) Visualization

## Introduction of the t-SNE visualization tool

MMPretrain provides `tools/visualization/vis_tsne.py` tool to visualize ... by t-SNE. Please install `sklearn` to calculate TSNE by `pip install scikit-learn`.

**Command**：

```bash
python tools/visualization/vis_tsne.py \
    CONFIG \
    [--checkpoint CHECKPOINT] \
    [--work-dir WORK_DIR] \
    [--test-cfg TEST_CFG] \
    [--vis-stage {backbone,neck,pre_logits}]
    [--class-idx ${CLASS_IDX} [CLASS_IDX ...]]
    [--max-num-class MAX_NUM_CLASS]
    [--max-num-samples MAX_NUM_SAMPLES]
    [--cfg-options CFG_OPTIONS [CFG_OPTIONS ...]]
    [--device DEVICE]
    [--legend]
    [--show]
    [--n-components N_COMPONENTS]
    [--perplexity PERPLEXITY]
    [--early-exaggeration EARLY_EXAGGERATION]
    [--learning-rate LEARNING_RATE]
    [--n-iter N_ITER]
    [--n-iter-without-progress N_ITER_WITHOUT_PROGRESS]
    [--init INIT]
```

**Description of all arguments**：

- `CONFIG`: The path of t-SNE config file.
- `--checkpoint CHECKPOINT`: The path of the checkpoint file.
- `--work-dir WORK_DIR`: The dir to save logs and models.
- `--test-cfg TEST_CFG`: The path of t-SNE config file to load config of test dataloader.
- `--vis-stage {backbone,neck,pre_logits}`: The visualization stage of the model.
- `--class-idx CLASS_IDX [CLASS_IDX ...]`: The categories used to calculate t-SNE.
- `--max-num-class MAX_NUM_CLASS`: The first N categories to apply t-SNE algorithms. Defaults to 20.
- `--max-num-samples MAX_NUM_SAMPLES`: The maximum number of samples per category. Higher number need longer time to calculate. Defaults to 100.
- `--cfg-options CFG_OPTIONS [CFG_OPTIONS ...]`: override some settings in the used config, the key-value pair in xxx=yyy format will be merged into config file. If the value to be overwritten is a list, it should be like key="[a,b]" or key=a,b It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" Note that the quotation marks are necessary and that no white space is allowed.
- `--device DEVICE`: Device used for inference.
- `--legend`: Show the legend of all categories.
- `--show`: Display the result in a graphical window.
- `--n-components N_COMPONENTS`: The dimension of results.
- `--perplexity PERPLEXITY`: The perplexity is related to the number of nearest neighbors that is used in other manifold learning algorithms.
- `--early-exaggeration EARLY_EXAGGERATION`: Controls how tight natural clusters in the original space are in the embedded space and how much space will be between them.
- `--learning-rate LEARNING_RATE`: The learning rate for t-SNE is usually in the range[10.0, 1000.0]. If the learning rate is too high, the data may looklike a ball with any point approximately equidistant from its nearestneighbours. If the learning rate is too low, most points may lookcompressed in a dense cloud with few outliers.
- `--n-iter N_ITER`: Maximum number of iterations for the optimization. Should be at least 250.
- `--n-iter-without-progress N_ITER_WITHOUT_PROGRESS`: Maximum number of iterations without progress before we abort the optimization.
- `--init INIT`: The init method.

## How to visualize the t-SNE of a CNN classifier (ResNet-18 and ResNet-50)

Here are two examples of running t-SNE visualization on ResNet-18 and ResNet-50 models, trained on CIFAR-10 dataset:

```shell
python tools/visualization/vis_tsne.py \
    configs/resnet/resnet18_8xb16_cifar10.py \
    --checkpoint https://download.openmmlab.com/mmclassification/v0/resnet/resnet18_b16x8_cifar10_20210528-bd6371c8.pth

python tools/visualization/vis_tsne.py \
    configs/resnet/resnet50_8xb16_cifar10.py \
    --checkpoint https://download.openmmlab.com/mmclassification/v0/resnet/resnet50_b16x8_cifar10_20210528-f54bfad9.pth
```

| ResNet-18                                                                                            | ResNet-50                                                                                            |
| ---------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------- |
| <div align=center><img src='https://user-images.githubusercontent.com/42371271/236410521-c4d087da-d16f-48ad-b951-c74d10c68f33.png' height="auto" width="auto" ></div> | <div align=center><img src='https://user-images.githubusercontent.com/42371271/236411844-c97dc514-dad0-401e-ba8f-307d0a385b4e.png' height="auto" width="auto" ></div> |

## How to visualize the t-SNE of a self-supervised vision transformer

To be added.
