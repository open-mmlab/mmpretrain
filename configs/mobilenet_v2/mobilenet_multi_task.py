_base_ = [
    '../_base_/datasets/multi_task_medic_data.py',
    '../_base_/models/mobilenet_v2_1x_multitask.py',
    '../_base_/schedules/imagenet_bs256_epochstep.py',
    '../_base_/default_runtime.py'
]
