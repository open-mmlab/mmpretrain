# Copyright (c) OpenMMLab. All rights reserved.
"""Create COCO-Style GT annotations based on raw annotation of Flickr30k.

GT annotations are used for evaluation in image caption task.
"""

import json


def main():
    with open('dataset_flickr30k.json', 'r') as f:
        annotations = json.load(f)
    ann_list = []
    img_list = []
    splits = ['train', 'val', 'test']
    for split in splits:
        for img in annotations['images']:

            # img_example={
            #     "sentids": [0, 1, 2],
            #     "imgid": 0,
            #     "sentences": [
            #         {"raw": "Two men in green shirts standing in a yard.",
            #          "imgid": 0, "sentid": 0},
            #         {"raw": "A man in a blue shirt standing in a garden.",
            #          "imgid": 0, "sentid": 1},
            #         {"raw": "Two friends enjoy time spent together.",
            #          "imgid": 0, "sentid": 2}
            #     ],
            #     "split": "train",
            #     "filename": "1000092795.jpg"
            # },

            if img['split'] != split:
                continue

            img_list.append({'id': img['imgid']})

            for sentence in img['sentences']:
                ann_info = {
                    'image_id': img['imgid'],
                    'id': sentence['sentid'],
                    'caption': sentence['raw']
                }
                ann_list.append(ann_info)

        json_file = {'annotations': ann_list, 'images': img_list}

        # generate flickr30k_train_gt.json, flickr30k_val_gt.json
        # and flickr30k_test_gt.json
        with open(f'flickr30k_{split}_gt.json', 'w') as f:
            json.dump(json_file, f)


if __name__ == '__main__':
    main()
