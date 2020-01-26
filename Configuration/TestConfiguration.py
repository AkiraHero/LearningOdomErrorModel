#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/08
# @Author  : Akira Ju
# @Email   : jukaka@pku.edu.cn
# @File    : TestConfiguration.py

"""
Combine different groups of settings for testing in machine learning.
"""

from ._Configuration import Configuration
from .SettingInstances import TestSetting, DataSetting, TestLogSetting, ModelSetting


class TestConfiguration(Configuration):
    def __init__(self, filepath=None, reinit=False):
        super(TestConfiguration, self).__init__(filepath, reinit)

    def display_key_setting(self):
        print('======Configurations======')
        print('Test Process Begins as following:')
        print('[DataSetName]', self.dataset.dataset_name, '::', self.dataset.branch, '-', self.dataset.imagesize)
        print('[NetWorkClass]', self.model.networkname)
        print('[Training Process]')
        print('\ttrainlogsuffix', self.logger.traininglogprefix)
        print('\tBatchSize', self.training.batch_size)

    def __init_setting_section__(self):
        self.testing = TestSetting()
        self.logger = TestLogSetting()
        self.dataset = DataSetting()
        self.model = ModelSetting()
