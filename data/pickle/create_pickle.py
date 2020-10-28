import os


from mmcls.datasets.persistences.persist_pickle import persist_imgs, persist_meta

# pickle类型的数据格式仅支持小数据集(图片量少于１w张时)，数据太大，模型训练时内存会溢出。
def main():
    img_dir = '/home/yanghui/yanghui/openset_v2/dataset/train_data/others'
    img_lst, gt_lst = [], []

    label_name_set = set()

    for idx, item_class in enumerate(os.listdir(img_dir)):
        class_path = '{}/{}'.format(img_dir, item_class)
        label_name_set.add(item_class)

        for item_img in os.listdir(class_path):
            img_path = '{}/{}'.format(class_path, item_img)
            img_lst.append(img_path)
            gt_lst.append(idx)

    persist_imgs(10, img_lst, gt_lst, (256, 256), 'training', './data/pickle')
    persist_meta(label_name_set, 'training', './data/pickle')

if __name__ == "__main__":
    main()