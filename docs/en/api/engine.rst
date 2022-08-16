.. role:: hidden
    :class: hidden-section

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

.. currentmodule:: mmcls.engine

Hook
------------------

.. autosummary::
   :toctree: generated
   :nosignatures:
   :template: classtemplate.rst

.. autoclass:: ClassNumCheckHook
.. autoclass:: PreciseBNHook
.. autoclass:: VisualizationHook

Optimizers
------------------

.. autosummary::
   :toctree: generated
   :nosignatures:
   :template: classtemplate.rst

.. autoclass:: Lamb
