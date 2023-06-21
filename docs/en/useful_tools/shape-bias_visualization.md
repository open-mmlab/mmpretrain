## Visualize Shape Bias

Shape bias measures how a model relies the shapes, compared to texture, to sense the semantics in images. For more details,
we recommend interested readers to this [paper](https://arxiv.org/abs/2106.07411). MMPretrain provide an off-the-shelf toolbox to
obtain the shape bias of a classification model. You can following these steps below:

### Prepare the dataset

First you should download the [cue-conflict](https://github.com/bethgelab/model-vs-human/releases/download/v0.1/cue-conflict.tar.gz) to `data` folder,
and then unzip this dataset. After that, you `data` folder should have the following structure:

```text
data
├──cue-conflict
|      |──airplane
|      |──bear
|      ...
|      |── truck
```
