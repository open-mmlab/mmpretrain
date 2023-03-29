Welcome to MMPretrain's documentation!
============================================

MMPretrain has set out to provide multiple powerful pre-trained backbones and
support different pre-training strategies. During the pre-training stage, we
anticipate the backbones to learn discriminative and rich semantic
representations of data. With the powerful pre-trained model, we are currently
capable of improving various downstream vision tasks.

Our primary objective for the codebase is to become an easily accessible and
user-friendly library. We aim to streamline research and engineering by
detailing the properties and design of MMPretrain across different sections.

Hands-on Roadmap of MMPretrain
-------------------------------

To help users quickly utilize MMPretrain, we recommend following the hands-on
roadmap we have created for the library:

   For users who want to try MMPretrain, we suggest reading the **Get Started**
   section for the environment setup.

   For basic usage, we refer users to **User Guides** for utilizing various
   algorithms to obtain the pre-trained models and evaluate their performance
   in downstream tasks.

   For those who wish to customize their own algorithms, we provide
   **Advanced Guides** that include hints and rules for modifying code.

   To find your desired pre-trained models, users could check the **Model Zoo**,
   which features a summary of various backbones and pre-training methods and
   introfuction of different algorithms.

   Additionally, we provide **Analysis** and **Visualization** tools to help
   diagnose algorithms.

   Besides, if you have any other questions or concerns, please refer to the 
   **Notes** section for potential answers.
   you may find answers.

We always welcome *PRs* and *Issues* for the betterment of MMPretrain.


.. toctree::
   :maxdepth: 1
   :caption: Get Started

   get_started.md

.. toctree::
   :maxdepth: 1
   :caption: User Guides

   user_guides/config.md
   user_guides/dataset_prepare.md
   user_guides/inference.md
   user_guides/train.md
   user_guides/test.md
   user_guides/downstream.md

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

.. toctree::
   :maxdepth: 1
   :caption: Model Zoo
   :glob:

   modelzoo_statistics.md
   papers/*

.. toctree::
   :maxdepth: 1
   :caption: Visualization

   useful_tools/dataset_visualization.md
   useful_tools/scheduler_visualization.md
   useful_tools/cam_visualization.md

.. toctree::
   :maxdepth: 1
   :caption: Analysis Tools

   useful_tools/print_config.md
   useful_tools/verify_dataset.md
   useful_tools/log_result_analysis.md
   useful_tools/complexity_analysis.md

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

.. toctree::
   :maxdepth: 1
   :caption: Notes

   notes/contribution_guide.md
   notes/projects.md
   notes/changelog.md
   notes/faq.md
   notes/pretrain_custom_dataset.md
   notes/finetune_custom_dataset.md

Indices and tables
==================

* :ref:`genindex`
* :ref:`search`
