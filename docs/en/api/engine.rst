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

Hooks
------------------

.. autosummary::
   :toctree: generated
   :nosignatures:
   :template: classtemplate.rst

   ClassNumCheckHook
   PreciseBNHook
   VisualizationHook

Optimizers
------------------

.. autosummary::
   :toctree: generated
   :nosignatures:
   :template: classtemplate.rst

   Lamb
