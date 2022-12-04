_base_ = ['./l-384-arc-rounda1.py']

_base_.train_dataloader.dataset.ann_file = 'meta/roundb1/train.txt'

model = dict(head=dict(ann_file='./data/ACCV_workshop/meta/roundb1/train.txt'))
