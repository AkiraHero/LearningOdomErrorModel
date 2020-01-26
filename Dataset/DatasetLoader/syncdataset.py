#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/04
# @Author  : Akira Ju
# @Email   : jukaka@pku.edu.cn
# @File    : syncdataset.py

import pandas as pd
import numpy as np


class SyncDataset():
    """
    Class SyncDataset: to load single frame data from a long sequence, components in any frame are synchronized.
    """
    def __init__(self, frameset=None):
        """
        Construction function.
        :param frameset: Set a frame window to crop the whole dataset.
        """
        self.croppedLength = 0
        self.dataArrayDict = {}
        self.otherDataLoaderDict = {}
        self.iniDataSetLenth = 0
        self.frameWindow = None
        self.dataTypeNum = 0
        if frameset != None:
            assert frameset[1] > frameset[0]
            self.croppedLength = frameset[1] - frameset[0]
            self.frameWindow = frameset



    def addSection(self, datafile, dataname, header=None):
        """
        Add a new  component for this sequence dataset.
        :param datafile: *.csv. Require no header, delimeter = ',', firstCol = timestamp/index
        :param dataname: variable name.
        :param header: using default None.
        :return: Bool. success flag.
        """
        try:
            dataself = pd.read_csv(datafile, header=header, delimiter=',').values
        except FileNotFoundError:
            return False

        #check section length
        if len(self.dataArrayDict):
            assert self.iniDataSetLenth == dataself.shape[0]
        else:
            self.iniDataSetLenth = dataself.shape[0]
            if not self.frameWindow:
                self.croppedLength = self.iniDataSetLenth

        if self.frameWindow:
            self.dataArrayDict[dataname] = dataself[self.frameWindow[0]:self.frameWindow[1], :]
        else:
            self.dataArrayDict[dataname] = dataself
        self.dataTypeNum += 1
        return True


    def registerDataLoader(self, loader, dataname):
        """
        Register other non-csv data loader such as image loader, binary loader
            Require DataLoader have these funtions:
                1.__len__
                2.__getitem__
                3.setFrameCropWin(frmwindow)
        :param loader:
        :param dataname:
        :return:
        """
        self.otherDataLoaderDict[dataname] = loader
        self.dataTypeNum += 1

        #check section length
        if len(self.dataArrayDict):
            assert self.iniDataSetLenth == len(loader)
        else:
            self.iniDataSetLenth = len(loader)

        if self.frameWindow:
            loader.setFrameCropWin(self.frameWindow)
        return True

    def timeStampCheck(self, baseDataName, dataname, errLimit):
        """
        Check whether different components of this sequence are sychronized.
        Require all timestamp are double(unit: second)
        :param baseDataName:
        :param dataname:
        :param errLimit:
        :return: list of index of unsynchronized data
        """
        assert self.dataArrayDict.get(baseDataName) != None
        assert self.dataArrayDict.get(dataname) != None
        baseTimestamp = self.dataArrayDict.get(baseDataName)[:, 0]
        timestamp = self.dataArrayDict.get(dataname)[:, 0]
        residual = baseTimestamp - timestamp
        overLimitInx = np.where(residual > errLimit)
        print("Time Sync Err DataNum:",len(overLimitInx))
        return len(overLimitInx)

    def __len__(self):
        return self.croppedLength

    def __getitem__(self, item_):
        sample = {}
        for i in self.dataArrayDict.items():
            name = i[0]
            datainfo = i[1][item_, :]
            sample[name] = datainfo

        for loader in self.otherDataLoaderDict.items():
            name = loader[0]
            sample[name] = loader[1][item_]

        return sample


