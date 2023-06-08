.. role:: hidden
    :class: hidden-section

.. module:: mmpretrain.evaluation

mmpretrain.evaluation
===================================

This package includes metrics and evaluators for classification tasks.

.. contents:: mmpretrain.evaluation
   :depth: 1
   :local:
   :backlinks: top

Single Label Metric
----------------------

.. autosummary::
   :toctree: generated
   :nosignatures:

   Accuracy
   SingleLabelMetric
   ConfusionMatrix

Multi Label Metric
----------------------
.. autosummary::
   :toctree: generated
   :nosignatures:

   AveragePrecision
   MultiLabelMetric
   VOCAveragePrecision
   VOCMultiLabelMetric

Retrieval Metric
----------------------

.. autosummary::
   :toctree: generated
   :nosignatures:
   :template: classtemplate.rst

   RetrievalRecall
   RetrievalAveragePrecision
