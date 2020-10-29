import lmdb
import numpy as np

from .builder import DATASETS
from .base_dataset import BaseDataset


@DATASETS.register_module()
class LMDBDataset(BaseDataset):

    def read_txt(self):
        data_infos = []
        with open(self.ann_file) as f:
            samples = [x.strip().split(' ') for x in f.readlines()]
            for filename, gt_label in samples:
                info = {'img_prefix': self.data_prefix}
                info['img_info'] = {'filename': filename}
                info['gt_label'] = np.array(gt_label, dtype=np.int64)
                data_infos.append(info)
            return data_infos

    def read_lmdb(self):
        data_infos = []
        env = lmdb.open(self.ann_file)
        txn = env.begin(write=False)
        for key, _ in txn.cursor():
            key = key.decode()
            if '###' not in key:
                continue

            label = float(key.split('###')[-1])
            data_infos.append({'img_info': {'filename': key},
                               'gt_label': np.array(label, dtype=np.int64),
                               'img_prefix': None})
        return data_infos

    def load_annotations(self):
        assert isinstance(self.ann_file, str)

        if self.ann_file.endswith('lmdb'):
            return self.read_lmdb()
        else:
            return self.read_txt()
