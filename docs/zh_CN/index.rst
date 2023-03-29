欢迎来到 MMPretrain 中文教程！
==========================================

MMPretrain 旨在提供各种强大的预训练主干网络，并支持了不同的预训练策略。在预训练阶段，
我们期望主干网络能够学习到更加具有判别性和丰富语义的数据表征。配备强大的预训练模型，
我们目前能够改善各种下游视觉任务。

我们的代码库旨在成为一个易于使用和用户友好的代码库库。为了帮助各类学术研究活动和工程任务，
我们在以下不同部分中详细介绍了 MMPretrain 的特性和设计。

MMPretrain 上手路线
-------------------------------

为了用户能够快速上手，我们推荐以下流程：

   对于想要使用 MMPretrain 的用户，我们推荐先阅读 **开始你的第一步** 部分来设置环境。

   对于一些基础使用，我们建议用户阅读 **教程** 来学习如何使用算法库来获得预训练模型以及在下游任务进行评测。

   若您想进行算法的自定义，我们提供了 **进阶教程** 来阐述了代码修改的方法和规则。

   如果您想找到所期望的预训练模型，您可以浏览 **模型库**，其中包含了模型库的总结，以及各类主干网络和预训练算法。
    
   我们同样提供了 **分析工具** 和 **可视化** 来辅助模型分析。

   另外，如果您还有其它问题，欢迎查阅 **其他说明**，也许可以找到您想要的答案。

非常欢迎用户的 PRs 和 Issues！

.. toctree::
   :maxdepth: 1
   :caption: 开始你的第一步

   get_started.md

.. toctree::
   :maxdepth: 1
   :caption: 教程

   user_guides/config.md
   user_guides/dataset_prepare.md
   user_guides/inference.md
   user_guides/train.md
   user_guides/test.md
   user_guides/downstream.md

.. toctree::
   :maxdepth: 1
   :caption: 进阶教程

   advanced_guides/datasets.md
   advanced_guides/pipeline.md
   advanced_guides/modules.md
   advanced_guides/schedule.md
   advanced_guides/runtime.md
   advanced_guides/evaluation.md
   advanced_guides/convention.md

.. toctree::
   :maxdepth: 1
   :caption: 模型库
   :glob:

   modelzoo_statistics.md
   papers/*

.. toctree::
   :maxdepth: 1
   :caption: 可视化

   useful_tools/dataset_visualization.md
   useful_tools/scheduler_visualization.md
   useful_tools/cam_visualization.md

.. toctree::
   :maxdepth: 1
   :caption: 分析工具

   useful_tools/print_config.md
   useful_tools/verify_dataset.md
   useful_tools/log_result_analysis.md
   useful_tools/complexity_analysis.md

.. toctree::
   :maxdepth: 1
   :caption: 部署

   useful_tools/model_serving.md

.. toctree::
   :maxdepth: 1
   :caption: 迁移指南

   migration.md

.. toctree::
   :maxdepth: 1
   :caption: API 参考文档

   mmpretrain.apis <api/apis>
   mmpretrain.engine <api/engine>
   mmpretrain.datasets <api/datasets>
   数据处理 <api/data_process>
   mmpretrain.models <api/models>
   mmpretrain.structures <api/structures>
   mmpretrain.visualization <api/visualization>
   mmpretrain.evaluation <api/evaluation>
   mmpretrain.utils <api/utils>

.. toctree::
   :maxdepth: 1
   :caption: 其他说明

   notes/contribution_guide.md
   notes/projects.md
   notes/changelog.md
   notes/faq.md
   notes/pretrain_custom_dataset.md
   notes/finetune_custom_dataset.md

.. toctree::
   :caption: 切换语言

   English <https://mmclassification.readthedocs.io/en/1.x/>
   简体中文 <https://mmclassification.readthedocs.io/zh_CN/1.x/>


索引与表格
==================

* :ref:`genindex`
* :ref:`search`
