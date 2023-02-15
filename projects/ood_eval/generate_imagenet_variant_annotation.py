import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument(
    '--imagenet1k-ann-file',
    type=str,
    help='path to the ImageNet1k annotation file')
parser.add_argument(
    '--imagenet-variant-root',
    type=str,
    help='the root folder of ImageNet variant')
parser.add_argument(
    '--imagenet-variant-name',
    type=str,
    help='the name of the ImageNet variant')
parser.add_argument(
    '--output-file', type=str, help='path to the output annotation file')

if __name__ == '__main__':
    args = parser.parse_args()
    with open(args.imagenet1k_ann_file, 'r') as f:
        imagenet1k_list = [line.strip().split() for line in f.readlines()]
        imagenet1k_list = [[line[0].split('/')[0], line[1]]
                           for line in imagenet1k_list]
    imagenet1k_label_map = {line[0]: line[1] for line in imagenet1k_list}

    imagenet_variant_images = []
    if args.imagenet_variant_name != 'c':
        # ImageNet variant A, R, S
        imagenet_variant_subfolders = os.listdir(args.imagenet_variant_root)
        imagenet_variant_subfolders = [subfolder for subfolder in imagenet_variant_subfolders if not subfolder.endswith('.txt')]
        for subfolder in imagenet_variant_subfolders:
            cur_label = imagenet1k_label_map[subfolder]
            cur_subfolder = os.path.join(args.imagenet_variant_root, subfolder)
            cur_subfolder_files = os.listdir(cur_subfolder)
            cur_subfolder_files = [
                os.path.join(subfolder, file) + ' ' + cur_label
                for file in cur_subfolder_files
            ]
            imagenet_variant_images.extend(cur_subfolder_files)
    else:
        # ImageNet variant C
        curruption_categories = os.listdir(args.imagenet_variant_root)
        for category in curruption_categories:
            curruption_levels = os.listdir(
                os.path.join(args.imagenet_variant_root, category))
            for level in curruption_levels:
                imagenet_variant_subfolders = os.listdir(
                    os.path.join(args.imagenet_variant_root, category, level))
                for subfolder in imagenet_variant_subfolders:
                    cur_label = imagenet1k_label_map[subfolder]
                    cur_subfolder = os.path.join(args.imagenet_variant_root,
                                                 category, level, subfolder)
                    cur_subfolder_files = os.listdir(cur_subfolder)
                    cur_subfolder_files = [
                        os.path.join(category, level, subfolder, file) + ' ' +
                        cur_label for file in cur_subfolder_files
                    ]
                    imagenet_variant_images.extend(cur_subfolder_files)

    with open(args.output_file, 'w') as f:
        f.write('\n'.join(imagenet_variant_images))
