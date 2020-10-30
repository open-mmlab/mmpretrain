import glob
import os
import re
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from typing import Tuple

import cv2
import lmdb
from loguru import logger

_10TB = 10 * (1 << 40)


class LmdbDataExporter(object):
    """
    making LMDB database
    """
    label_pattern = re.compile(r'/.*/.*?(\d+)$')

    def __init__(self,
                 img_dir=None,
                 output_path=None,
                 shape=(256, 256),
                 batch_size=100):
        """
            img_dir: imgs directory
            output_path: LMDB output path
        """
        self.img_dir = img_dir
        self.output_path = output_path
        self.shape = shape
        self.batch_size = batch_size
        self.label_list = list()

        if not os.path.exists(img_dir):
            raise Exception(f'{img_dir} is not exists!')

        if not os.path.exists(output_path):
            os.makedirs(output_path)

        # 最大10T
        self.lmdb_env = lmdb.open(output_path, map_size=_10TB, max_dbs=4)

        self.label_dict = defaultdict(int)

    def export(self):
        idx = 0
        results = []
        st = time.time()
        iter_img_lst = self.read_imgs()
        while True:
            items = []
            try:
                while len(items) < self.batch_size:
                    items.append(next(iter_img_lst))
            except StopIteration:
                break

            with ThreadPoolExecutor() as executor:
                results.extend(executor.map(self._extract_once, items))

            if len(results) >= self.batch_size:
                self.save_to_lmdb(results)
                idx += self.batch_size
                et = time.time()
                logger.info(f'time: {(et-st)}(s)  count: {idx}')
                st = time.time()
                del results[:]

        et = time.time()
        logger.info(f'time: {(et-st)}(s)  count: {idx}')
        self.save_to_lmdb(results)
        self.save_total(idx)
        del results[:]

    def save_to_lmdb(self, results):
        """
        persist to lmdb
        """
        with self.lmdb_env.begin(write=True) as txn:
            while results:
                img_key, img_byte = results.pop()
                if img_key is None or img_byte is None:
                    continue
                txn.put(img_key, img_byte)

    def save_total(self, total: int):
        """
        persist all numbers of imgs
        """
        with self.lmdb_env.begin(write=True, buffers=True) as txn:
            txn.put('total'.encode(), str(total).encode())

    def _extract_once(self, item) -> Tuple[bytes, bytes]:
        full_path = item[-1]
        imageKey = '###'.join(map(str, item[:-1]))

        img = cv2.imread(full_path)
        if img is None:
            logger.error(f'{full_path} is a bad img file.')
            return None, None
        if img.shape != self.shape:
            img = self.fillImg(img)
        _, img_byte = cv2.imencode('.jpg', img)
        return (imageKey.encode(), img_byte.tobytes())

    def fillImg(self, img):
        width = img.shape[1]
        height = img.shape[0]
        top, bottom, left, right = 0, 0, 0, 0
        if width > height:
            diff = width - height
            top = int(diff / 2)
            bottom = diff - top
        else:
            diff = height - width
            left = int(diff / 2)
            right = diff - left
        fimg = cv2.copyMakeBorder(
            img,
            top,
            bottom,
            left,
            right,
            cv2.BORDER_CONSTANT,
            value=[0, 0, 0])
        rimg = cv2.resize(fimg, self.shape, interpolation=cv2.INTER_AREA)
        return rimg

    def read_imgs(self):
        img_list = glob.glob(os.path.join(self.img_dir, '*/*.jpg'))

        for idx, item_img in enumerate(img_list):
            label = item_img.split('/')[-2]
            if label not in self.label_list:
                self.label_list.append(label)

            item = (idx, self.label_list.index(label), item_img)
            yield item
