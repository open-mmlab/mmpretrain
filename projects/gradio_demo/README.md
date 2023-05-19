# MMPretrain Gradio Demo

Here is a gradio demo for MMPretrain supported inference tasks.

Currently supported tasks:

- Image Classifiation
- Image-To-Image Retrieval
- Text-To-Image Retrieval (require multi-modality support)
- Image Caption (require multi-modality support)
- Visual Question Answering (require multi-modality support)
- Visual Grounding (require multi-modality support)

## Preview

<img src="https://user-images.githubusercontent.com/26739999/236147750-90ccb517-92c0-44e9-905e-1473677023b1.jpg" width="100%"/>

## Requirements

To run the demo, you need to install MMPretrain at first. And please install with the extra multi-modality
dependencies to enable multi-modality tasks.

```shell
# At the MMPretrain root folder
pip install -e ".[multimodal]"
```

And then install the latest gradio package.

```shell
pip install "gradio>=3.31.0"
```

## Start

Then, you can start the gradio server on the local machine by:

```shell
# At the project folder
python launch.py
```

The demo will start a local server `http://127.0.0.1:7860` and you can browse it by your browser.
And to share it to others, please set `share=True` in the `demo.launch()`.
