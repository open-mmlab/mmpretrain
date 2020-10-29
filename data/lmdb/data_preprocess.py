import os
import cv2
import click


@click.option('-i', 'img_dir', required=True, help='图片文件夹路径')
@click.option('-o', 'output_lst', required=True, help='lst文件输出路径')
@click.command()
def main(img_dir: str, output_lst: str):
    if img_dir.startswith('.'):
        raise ('请用绝对路径赋值img_dir变量')

    lst_str, img_num = '', 0

    for label, item_class in enumerate(os.listdir(img_dir)):
        print('preprocess class {} for all imgs {}'.format(
            item_class, img_num))
        class_path = '{}/{}'.format(img_dir, item_class)
        for item_img in os.listdir(class_path):
            abs_img_path = os.path.join(class_path, item_img)
            if cv2.imread(abs_img_path) is None:
                continue
            lst_str += '{}\t{}\t{}\n'.format(img_num, label, abs_img_path)
            img_num += 1

    open(output_lst, 'w').write(lst_str)


if __name__ == '__main__':
    main()
