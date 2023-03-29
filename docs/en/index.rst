Welcome to MMPretrain's documentation!
============================================

MMPretrain aims to provide various powerful pre-trained backbones and supports 
different pre-training strategies. In pre-training stage, we expect that 
backbones are able to learn the discriminative and rich semantic representation
of the data. Equipped with the powerful pre-trained model, we are able to 
improve various downstream vision tasks currently.

Our codebase aims to become an easy-to-use and user-friendly library. To help 
the research and engineering, we elaborate the properties and design of 
MMPretrain in the following different sections.


Hands-on Roadmap of MMPretrain
-------------------------------

To help the user to use the MMPretrain quickly, we recommend the following 
roadmap for using our library:

   For the user who wants to try MMPretrain, we recommend the user to read
   **Get Started** part for the environment setup.

   For the basic usage, we refer the user to **User Guides** for using various 
   algorithms to obtain the pre-trained models and evaulate them in downstream 
   tasks.

   For the customization of your own algorithms, we provide **Advanced Guides** 
   illustrating hints and rules of code modification.

   To find your desired pre-trained models, you could check the **Model Zoo**,
   which consists of summary, various backbones and pre-training methods.
    
   Also, we provide some **Analysis** and **Visualization** tools to help 
   diagnose the algorithms.

   Besides, if you have any other questions, please refer to **Notes**, where 
   you may find answers.

PRs and Issues are welcome!


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
