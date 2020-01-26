#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/04
# @Author  : Akira Ju
# @Email   : jukaka@pku.edu.cn
# @File    : seqDataset.py


from .syncdataset import SyncDataset
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from Utils.debugUtils import itemmap


class SeqDataset(SyncDataset, Dataset):
    """
    Class SeqDataset: to load subsequence data from a long sequence, components in any sequence are synchronized.
    """
    def __init__(self, seqLength, frmset=None, seglistfile=None):
        """
        Construction function.
        :param seqLength: Fixed sequence length of data items if seglistfile=None, else: max sequence length of data items.
        :param frmset: Set a frame window to crop the whole dataset.
        :param seglistfile: [optional] when specified, using sequence definition from outer file.
        """
        if seglistfile is None:
            super(SeqDataset, self).__init__(frameset=frmset)
            self.seqLen = seqLength
            self.using_seg_list = False
        else:
            super(SeqDataset, self).__init__()
            self.seg_checked = False
            self.seqLen = seqLength
            self.using_seg_list = True
            self.seglist = pd.read_csv(seglistfile, header=None).values
            if frmset:
                assert frmset[0] >= 0 and frmset[1] <= self.seglist.shape[0]
                self.seglist = self.seglist[frmset[0]:frmset[1], :]

    def __len__(self):
        if not self.using_seg_list:
            return self.croppedLength - self.seqLen + 1
        else:
            return self.seglist.shape[0]

    def __getitem__(self, item):
        assert item <= self.__len__()
        sample = {}
        if not self.using_seg_list:
            for t in range(self.seqLen):
                item_ = item + t
                singleSample = super(SeqDataset, self).__getitem__(item_)
                for p in singleSample.items():
                    name = p[0]
                    if not sample.get(name):
                        sample[name] = []
                    sample[name].append(p[1])
            return sample
        else:
            assert self.seg_checked
            seg = self.seglist[item]
            for t in range(self.seqLen):
                item_ = seg[0] + t
                if item_ < seg[1]:
                    singleSample = super(SeqDataset, self).__getitem__(item_)
                    for p in singleSample.items():
                        name = p[0]
                        if not sample.get(name):
                            sample[name] = []
                        sample[name].append(p[1])
                # to avoid customizing collate_fn for batch loader
                else:
                    for p in singleSample.items():
                        name = p[0]
                        sample[name].append(p[1] * 0)
                sample['seglength'] = seg[1] - seg[0]
            return sample

    def check_seg_data(self):
        """
        Check data sequences
        """
        # check the seglistfile
        assert self.seglist.shape[1] == 2
        assert np.max(self.seglist[:, 1]) <= self.croppedLength, "There are segs with too large inx !"
        assert np.max(self.seglist[:, 0]) >= 0, "There are segs with negative inx !"
        assert np.min(self.seglist[:, 1] - self.seglist[:, 0]) >= 1, "There are segs with length < 1!"
        maxseglen = np.max(self.seglist[:, 1] - self.seglist[:, 0])
        maxseginx = np.argmax(self.seglist[:, 1] - self.seglist[:, 0])
        maxseg = self.seglist[maxseginx]
        assert maxseglen <= self.seqLen, "There are segs {}-{} with max length:{} > time_step_num:{}!".format(maxseg[0], maxseg[1], maxseglen, self.seqLen)
        self.seg_checked = True

