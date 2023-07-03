Welcome to MMPretrain's documentation!
============================================

MMPretrain is a newly upgraded open-source framework for pre-training.
It has set out to provide multiple powerful pre-trained backbones and
support different pre-training strategies. MMPretrain originated from the
famous open-source projects
`MMClassification <https://github.com/open-mmlab/mmclassification/tree/1.x>`_
and `MMSelfSup <https://github.com/open-mmlab/mmselfsup>`_, and is developed
with many exiciting new features. The pre-training stage is essential for
vision recognition currently. With the rich and strong pre-trained models,
we are currently capable of improving various downstream vision tasks.

Our primary objective for the codebase is to become an easily accessible and
user-friendly library and to streamline research and engineering. We
detail the properties and design of MMPretrain across different sections.

Hands-on Roadmap of MMPretrain
-------------------------------

To help users quickly utilize MMPretrain, we recommend following the hands-on
roadmap we have created for the library:

   - For users who want to try MMPretrain, we suggest reading the GetStarted_
     section for the environment setup.

   - For basic usage, we refer users to UserGuides_ for utilizing various
     algorithms to obtain the pre-trained models and evaluate their performance
     in downstream tasks.

   - For those who wish to customize their own algorithms, we provide
     AdvancedGuides_ that include hints and rules for modifying code.

   - To find your desired pre-trained models, users could check the ModelZoo_,
     which features a summary of various backbones and pre-training methods and
     introfuction of different algorithms.

   - Additionally, we provide Analysis_ and Visualization_ tools to help
     diagnose algorithms.

   - Besides, if you have any other questions or concerns, please refer to the
     Notes_ section for potential answers.

We always welcome *PRs* and *Issues* for the betterment of MMPretrain.

.. _GetStarted:
.. toctree::
   :maxdepth: 1
   :caption: Get Started

   get_started.md

.. _UserGuides:
.. toctree::
   :maxdepth: 1
   :caption: User Guides

   user_guides/config.md
   user_guides/dataset_prepare.md
   user_guides/inference.md
   user_guides/train.md
   user_guides/test.md
   user_guides/downstream.md

.. _AdvancedGuides:
.. toctree::
   :maxdepth: 1
   :caption: Advanced Guides

   advanced_guides/datasets.md
   advanced_guides/pipeline.md
   advanced_guides/modules.md
   advanced_guides/schedule.md
   advanced_guides/runtime.md
   advanced_guides/evaluation.md
   advanced_guides/convention.md

.. _ModelZoo:
.. toctree::
   :maxdepth: 1
   :caption: Model Zoo
   :glob:

   modelzoo_statistics.md
   papers/*

.. _Visualization:
.. toctree::
   :maxdepth: 1
   :caption: Visualization

   useful_tools/dataset_visualization.md
   useful_tools/scheduler_visualization.md
   useful_tools/cam_visualization.md
   useful_tools/t-sne_visualization.md

.. _Analysis:
.. toctree::
   :maxdepth: 1
   :caption: Analysis Tools

   useful_tools/print_config.md
   useful_tools/verify_dataset.md
   useful_tools/log_result_analysis.md
   useful_tools/complexity_analysis.md
   useful_tools/confusion_matrix.md
   useful_tools/shape_bias.md

.. toctree::
   :maxdepth: 1
   :caption: Deployment

   useful_tools/model_serving.md

.. toctree::
   :maxdepth: 1
   :caption: Migration

   migration.md

.. toctree::
   :maxdepth: 1
   :caption: API Reference

   mmpretrain.apis <api/apis>
   mmpretrain.engine <api/engine>
   mmpretrain.datasets <api/datasets>
   Data Process <api/data_process>
   mmpretrain.models <api/models>
   mmpretrain.structures <api/structures>
   mmpretrain.visualization <api/visualization>
   mmpretrain.evaluation <api/evaluation>
   mmpretrain.utils <api/utils>

.. _Notes:
.. toctree::
   :maxdepth: 1
   :caption: Notes

   notes/contribution_guide.md
   notes/projects.md
   notes/changelog.md
   notes/faq.md
   notes/pretrain_custom_dataset.md
   notes/finetune_custom_dataset.md

.. toctree::
   :maxdepth: 1
   :caption: Device Support

   device/npu.md

Indices and tables
==================

* :ref:`genindex`
* :ref:`search`
