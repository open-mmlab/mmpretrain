import os
import re
import time
from typing import Tuple
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

import lmdb
import cv2
from loguru import logger

logger.add("./lmdb.log", level="INFO")
_10TB = 10 * (1 << 40)


class IdtLmdbDataExporter(object):
    """
    用于将小文件导出对应的lmdb文件
    """
    label_pattern = re.compile(r"/.*/.*?(\d+)$")

    def __init__(self,
                 target_path,
                 output_path=None,
                 shape=(256, 256),
                 batch_size=100):
        """
            target_path: list 路径(由cfim2rec.py生成的list文件)
            output_path: lmdb输出路径
        """
        self.target_path = target_path
        self.output_path = output_path
        self.shape = shape
        self.batch_size = batch_size
        self.class_num_dict = dict()

        if not os.path.exists(target_path):
            raise Exception(f"{target_path} is not exists!")

        # list文件列表
        self.files = [
            os.path.join(target_path, fname)
            for fname in os.listdir(target_path)
            if os.path.isfile(os.path.join(target_path, fname))
            and fname.endswith(".lst")
        ]
        # self.files = ["./rec-test/val.lst"]

        if not os.path.exists(output_path):
            os.makedirs(output_path)

        # 最大10T
        self.lmdb_env = lmdb.open(output_path, map_size=_10TB, max_dbs=4)

        self.label_dict = defaultdict(int)

    def export(self):
        results = []
        st = time.time()
        for file_lst in self.files:
            idx = 0
            logger.info(f'Creating lmdb file from {file_lst}')

            iter_file_lst = self.read_list(file_lst)
            while True:
                items = []
                try:
                    while len(items) < self.batch_size:
                        items.append(next(iter_file_lst))
                except StopIteration:
                    break

                with ThreadPoolExecutor() as executor:
                    results.extend(executor.map(self._extract_once, items))

                if len(results) >= self.batch_size:
                    self.save_to_lmdb(idx, results)
                    idx += self.batch_size
                    et = time.time()
                    logger.info(f"time: {(et-st)}(s)  count: {idx}")
                    st = time.time()
                    del results[:]

            et = time.time()
            logger.info(f"time: {(et-st)}(s)  count: {idx}")
            self.save_to_lmdb(idx, results)
            self.save_total(idx)
            del results[:]

    def save_to_lmdb(self, start_idx: int, results):
        """
        结果持久化到lmdb
        """
        with self.lmdb_env.begin(write=True) as txn:
            while results:
                img_key, img_byte = results.pop()
                if img_key is None or img_byte is None:
                    continue
                txn.put(str(start_idx).encode(), img_key)
                txn.put(img_key, img_byte)
                start_idx += 1

    def save_total(self, total: int):
        """
        持久化总记录
        """
        class_num = len(self.class_num_dict.keys())
        with self.lmdb_env.begin(write=True, buffers=True) as txn:
            txn.put("total".encode(), str(total).encode())
            txn.put("class_num".encode(), str(class_num).encode())
            for item_class in self.class_num_dict.keys():
                txn.put(f"class#{item_class}".encode(),
                        str(self.class_num_dict[item_class]).encode())

    def _extract_once(self, item) -> Tuple[bytes, bytes]:
        full_path = item[-1]
        imageKey = "###".join(map(str, item[:-1]))

        label = item[-2]
        if label in self.class_num_dict.keys():
            self.class_num_dict[label] += 1
        else:
            self.class_num_dict[label] = 1

        img = cv2.imread(full_path)
        if img is None:
            return None, None
        if img.shape != self.shape:
            img = self.fillImg(img)
        _, img_byte = cv2.imencode(".jpg", img)
        logger.debug(f"{imageKey} for {full_path}")
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

    def read_list(self, path_in):
        with open(path_in) as fin:
            while True:
                line = fin.readline()
                if not line:
                    break
                line = [i.strip() for i in line.strip().split('\t')]
                line_len = len(line)
                if line_len < 3:
                    logger.warning('lst should at least has three parts, '
                                   'but only has %s parts for %s' %
                                   (line_len, line))
                    continue

                item = (int(line[0]), float(line[1]), line[2])
                yield item
