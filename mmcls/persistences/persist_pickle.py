import os
import cv2
import pickle
import numpy as np


def persist_imgs(batch_num, imgs_lst, gt_lst, img_size, train_val_tag='training', persist_dir=''):
    assert len(imgs_lst) == len(gt_lst)
    imgs_num = len(imgs_lst)
    img_num_4_each_batch = int(imgs_num/batch_num)

    for item_batch in range(batch_num):
        print('persist {}th batch of {} imgs'.format(item_batch, img_num_4_each_batch))
        start_index = item_batch * img_num_4_each_batch
        last_index = start_index + img_num_4_each_batch
        last_index = imgs_num if imgs_num < last_index else last_index

        item_img_lst = imgs_lst[start_index: last_index]
        item_gt_lst = gt_lst[start_index: last_index]

        data, labels, filenames = [], [], []
        for (item_img, item_gt) in zip(item_img_lst, item_gt_lst):
            if item_img[-3:] not in ['jpg', 'png']:
                continue
            try:
                img = cv2.resize(cv2.imread(item_img), img_size)
                data.append(img)
                labels.append(item_gt)
                filenames.append(os.path.basename(item_img))
            except:
                print('{} persist error!'.format(item_img))
                continue
        
        if len(data) == 0:
            continue
        
        persist_dict = {'batch_label': '{} batch {} of {}'.format(train_val_tag, item_batch, batch_num),
                        'data': np.array(data),
                        'labels': labels,
                        'filsnames': filenames}

        persist_path = '{}/pickle-train-batches/data_batch_{}'.format(
            persist_dir, item_batch) if train_val_tag == 'training' else '{}/pickle-val-batches/val_batch_{}'.format(persist_dir, item_batch)
        pf = open(persist_path, 'wb')  # 注意一定要写明是wb 而不是w.
        pickle.dump(persist_dict, pf)
        pf.close()

def persist_meta(label_name_lst, train_val_tag='training', persist_dir=''):
    pf = open('{}/batches.meta'.format(persist_dir), 'wb')  # 注意一定要写明是wb 而不是w.
    pickle.dump(list(label_name_lst), pf)
    pf.close()
