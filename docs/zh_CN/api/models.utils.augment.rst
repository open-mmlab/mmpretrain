.. role:: hidden
    :class: hidden-section

Batch Augmentation
===================================

Batch augmentation is the augmentation which involve multiple samples, such as Mixup and CutMix.

In MMClassification, these batch augmentation is used as a part of :ref:`classifiers`. A typical usage is as below:

.. code-block:: python

   model = dict(
       backbone = ...,
       neck = ...,
       head = ...,
       train_cfg=dict(augments=[
           dict(type='BatchMixup', alpha=0.8, prob=0.5, num_classes=num_classes),
           dict(type='BatchCutMix', alpha=1.0, prob=0.5, num_classes=num_classes),
       ]))
   )

.. currentmodule:: mmcls.models.utils.augment

Mixup
-----
.. autoclass:: BatchMixupLayer

CutMix
------
.. autoclass:: BatchCutMixLayer

ResizeMix
---------
.. autoclass:: BatchResizeMixLayer
