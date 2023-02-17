.. role:: hidden
    :class: hidden-section

.. module:: mmcls.evaluation

mmcls.evaluation
===================================

This package includes metrics and evaluators for classification tasks.

.. contents:: mmcls.evaluation
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
<<<<<<< HEAD

.. autosummary::
   :toctree: generated
   :nosignatures:
   :template: classtemplate.rst

   RetrievalRecall
=======
.. autosummary::
   :toctree: generated
   :nosignatures:

   RetrievalAveragePrecision
>>>>>>> 544324a... refactor retrieval metric
