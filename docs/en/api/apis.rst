.. role:: hidden
    :class: hidden-section

.. module:: mmpretrain.apis

mmpretrain.apis
===================================

These are some high-level APIs for classification tasks.

.. contents:: mmpretrain.apis
   :depth: 2
   :local:
   :backlinks: top

Model
------------------

.. autosummary::
   :toctree: generated
   :nosignatures:

   list_models
   get_model

Inference
------------------

.. autosummary::
   :toctree: generated
   :nosignatures:
   :template: callable.rst

   ImageClassificationInferencer
   ImageRetrievalInferencer
   ImageCaptionInferencer
   VisualQuestionAnsweringInferencer
   VisualGroundingInferencer
   TextToImageRetrievalInferencer
   ImageToTextRetrievalInferencer
   NLVRInferencer
   FeatureExtractor

.. autosummary::
   :toctree: generated
   :nosignatures:

   inference_model
