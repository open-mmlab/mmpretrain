.. role:: hidden
    :class: hidden-section

.. module:: mmpretrain.engine

mmpretrain.engine
===================================

This package includes some runtime components, including hooks, runners, optimizers and loops. These components are useful in
classification tasks but not supported by MMEngine yet.

.. note::

   Some components may be moved to MMEngine in the future.

.. contents:: mmpretrain.engine
   :depth: 2
   :local:
   :backlinks: top

.. module:: mmpretrain.engine.hooks

Hooks
------------------

.. autosummary::
   :toctree: generated
   :nosignatures:

   ClassNumCheckHook
   PreciseBNHook
   VisualizationHook
   PrepareProtoBeforeValLoopHook
   SetAdaptiveMarginsHook
   EMAHook
   SimSiamHook
   DenseCLHook
   SwAVHook

.. module:: mmpretrain.engine.optimizers

Optimizers
------------------

.. autosummary::
   :toctree: generated
   :nosignatures:

   Lamb
   LARS
   LearningRateDecayOptimWrapperConstructor
