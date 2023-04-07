.. role:: hidden
    :class: hidden-section

.. module:: mmcls.engine

mmcls.engine
===================================

This package includes some runtime components, including hooks, runners, optimizers and loops. These components are useful in
classification tasks but not supported by MMEngine yet.

.. note::

   Some components may be moved to MMEngine in the future.

.. contents:: mmcls.engine
   :depth: 2
   :local:
   :backlinks: top

.. module:: mmcls.engine.hooks

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
   PushDataInfoToMessageHubHook

.. module:: mmcls.engine.optimizers

Optimizers
------------------

.. autosummary::
   :toctree: generated
   :nosignatures:

   Lamb
