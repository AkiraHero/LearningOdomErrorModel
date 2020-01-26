#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/12
# @Author  : Akira Ju
# @Email   : jukaka@pku.edu.cn
# @File    : genSegListFile.py

"""
Function:
    Pick subsequence from a long sequence orderly according to triggers like steering/travelling distance threshold
Usage:
    python enCapsulateImg2Bin.py imgDir [inxfile]
"""

import pandas as pd
import numpy as np
import sys
if __name__ == '__main__':
    gt_rel_pose_file = sys.argv[1]
    max_seg_dis = float(sys.argv[2]) #m
    seglist_file = sys.argv[1] + '.seg'
    gt_rel_pose = pd.read_csv(gt_rel_pose_file, header=None).values
    data_length = gt_rel_pose.shape[0]
    rel_dist_list = (gt_rel_pose[:, 1] ** 2 + gt_rel_pose[:, 2] ** 2) ** 0.5
    seglist = []
    for i in range(data_length):
        inx = i
        cur_dis = 0
        while cur_dis < max_seg_dis and inx < data_length:
            cur_dis += rel_dist_list[inx]
            inx += 1
        if cur_dis > max_seg_dis:
            if len(seglist) > 0 and seglist[-1][1] == inx:
                continue
            seglist.append([i, inx])
    with open(seglist_file, 'w') as f:
        for seg in seglist:
            s = "{:.0f},{:.0f}\n".format(seg[0], seg[1])
            f.write(s)

