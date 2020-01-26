#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/04
# @Author  : Akira Ju
# @Email   : jukaka@pku.edu.cn
# @File    : ImageTensorBinLoader.py

"""
A image loader for binary zipped images file "*.tbin".

"""

import torch
import os
import pandas as pd
import io


class ImageTensorBinLoader:
    def __init__(self, binfile, binInfofile, timefile=None, loadall2mem=True):
        """
        Construction function.
        :param binfile: binary file.
        :param binInfofile: binary information file.
        :param timefile: [optional] timestamp/inx file.
        :param loadall2mem: [optional] load all frames to memory if True, else using frame-by-frame mode.
        """
        self.binfile = binfile
        self.frameWindow = None
        self.binSize = os.path.getsize(binfile)
        self.buffersize = 0
        self.imgNum = 0
        self._getBinInfo(binInfofile)
        if timefile is not None:
            self.timeinfoList = pd.read_csv(timefile, header=None).values
            assert self.timeinfoList.shape[0] == self.imgNum
        self.binStream = open(binfile, 'rb')
        assert self.binStream.readable()
        assert self.binSize == self.imgNum * self.buffersize
        self.loadall2mem = loadall2mem
        if loadall2mem:
            self.buf = io.BytesIO(self.binStream.read(self.binSize))

    def __len__(self):
        if self.frameWindow == None:
            return self.imgNum
        else:
            return self.frameWindow[1] - self.frameWindow[0]
        return 0

    def __getitem__(self, item):
        if self.frameWindow:
            realItem = self.frameWindow[0] + item
        else:
            realItem = item
        if not self.loadall2mem:
            self.binStream = open(self.binfile, 'rb')
            self.binStream.seek(self.buffersize * realItem, os.SEEK_SET)
            buf = io.BytesIO(self.binStream.read(self.buffersize))
            img = torch.load(buf)
            self.binStream.close()
        else:
            self.buf.seek(self.buffersize * realItem, os.SEEK_SET)
            img = torch.load(self.buf)
        return img

    def setFrameCropWin(self, win):
        """
        Set a frame window, discard data out of this window.
        :param win: int pair (st, end)
        """
        assert win[1] > win[0]
        assert win[1] <= self.imgNum
        self.frameWindow = win

    def _getBinInfo(self, file):
        """
        Load information of this binary image dataset incluing buffersize of every image and total imgNum.
        :param file:
        """
        lines = open(file, 'r').readlines()
        if lines[0].split(':')[-1][-1] == '\n':
            self.buffersize = int(lines[0].split(':')[-1][:-1])
        else:
            self.buffersize = int(lines[0].split(':')[-1])
        if lines[1].split(':')[-1][-1] == '\n':
            self.imgNum = int(lines[1].split(':')[-1][:-1])
        else:
            self.imgNum = int(lines[1].split(':')[-1])