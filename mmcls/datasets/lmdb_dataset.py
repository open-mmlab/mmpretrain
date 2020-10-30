import glob
import os

import lmdb
import numpy as np

from .base_dataset import BaseDataset
from .builder import DATASETS


@DATASETS.register_module()
class LMDBDataset(BaseDataset):

    def read_lmdb(self):
        data_infos = []
        env = lmdb.open(self.ann_file)
        txn = env.begin(write=False)
        for key, _ in txn.cursor():
            key = key.decode()
            if '###' not in key:
                continue

            label = float(key.split('###')[-1])
            data_infos.append({
                'img_info': {
                    'filename': key
                },
                'gt_label': np.array(label, dtype=np.int64),
                'img_prefix': None
            })
        return data_infos

    def load_annotations(self):
        assert isinstance(self.ann_file, str)

        if os.path.isdir(self.ann_file):
            mdb_list = glob.glob(os.path.join(self.ann_file, '*.mdb'))
            if len(mdb_list) == 0:
                raise Exception(f'no mdb file in {self.ann_file}.')
            if len(mdb_list) != 2:
                raise Exception('num of mdb files must be 2.')
            return self.read_lmdb()
        else:
            raise Exception('ann_file path is not an lmdb folder.')
