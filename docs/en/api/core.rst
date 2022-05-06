.. role:: hidden
    :class: hidden-section

mmcls.core
===================================

This package includes some runtime components. These components are useful in
classification tasks but not supported by MMCV yet.

.. note::

   Some components may be moved to MMCV in the future.

.. contents:: mmcls.core
   :depth: 2
   :local:
   :backlinks: top

.. currentmodule:: mmcls.core

Evaluation
------------------

Evaluation metrics calculation functions

.. autosummary::
   :toctree: generated
   :nosignatures:

   precision
   recall
   f1_score
   precision_recall_f1
   average_precision
   mAP
   support
   average_performance
   calculate_confusion_matrix

Hook
------------------

.. autosummary::
   :toctree: generated
   :nosignatures:
   :template: classtemplate.rst

   ClassNumCheckHook
   PreciseBNHook
   CosineAnnealingCooldownLrUpdaterHook
   MMClsWandbHook


Optimizers
------------------

.. autosummary::
   :toctree: generated
   :nosignatures:
   :template: classtemplate.rst

   Lamb
