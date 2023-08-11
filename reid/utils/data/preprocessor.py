from __future__ import absolute_import
import os.path as osp

from PIL import Image
from collections import defaultdict


class Preprocessor(object):
    def __init__(self, dataset, root=None, transform=None):
        super(Preprocessor, self).__init__()
        self.dataset = dataset
        self.root = root
        self.transform = transform

        self.index_pid = defaultdict(int)
        self.pid_cam = defaultdict(list)
        self.pid_index = defaultdict(list)
        for index, (_, pid, cam, _) in enumerate(dataset):  # pid = img_path
            if pid < 0:  # max(index) = len(data_source)
                continue
            self.index_pid[index] = pid  # index_pid means the  `pid` corresponding to the `index` . 1->1
            self.pid_cam[pid].append(cam)  # pid_cam means the  `cam_id` corresponding to the `pid`. 1->list
            self.pid_index[pid].append(index)  # pid_index means the `index` corresponding to the `pid`. 1->list


    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, indices):
        if isinstance(indices, (tuple, list)):
            return [self._get_single_item(index) for index in indices]
        return self._get_single_item(indices)

    def _get_single_item(self, index):
        fname, pid, camid, _ = self.dataset[index]
        fpath = fname
        if self.root is not None:
            fpath = osp.join(self.root, fname)
        img = Image.open(fpath).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img, fname, pid, sorted(set(self.pid_cam[pid])).index(camid)
