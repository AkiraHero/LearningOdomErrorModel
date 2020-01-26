#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/08
# @Author  : Akira Ju
# @Email   : jukaka@pku.edu.cn
# @File    : TrainingConfiguration.py

"""
Combine different groups of settings for training in machine learning.
"""

from ._Configuration import Configuration
from .SettingInstances import TrainingSetting, DataSetting, ModelSetting, LogSetting


class TrainingConfiguration(Configuration):
    def __init__(self, filepath=None, reinit=False):
        super(TrainingConfiguration, self).__init__(filepath, reinit)

    def display_key_setting(self):
        print('======Configurations======')
        print('Training Process Begins as following:')
        print('[DataSetName]', self.dataset.dataset_name, '::', self.dataset.branch, '-', self.dataset.image_size)
        print('[NetWorkClass]', self.model.classname)
        print('[Training Process]')
        print('\tEpochNum',self.training.epoch)
        print('\tBatchSize', self.training.batch_size)
        print('\tLearningRate', self.training.learning_rate)
        print('\tTimestepNum', self.training.time_step_num)
        print('\tOptimizer', self.training.optimizer)
        print('\tShuffle', self.training.shuffle)

    def __init_setting_section__(self):
        self.training = TrainingSetting()
        self.model = ModelSetting()
        self.logger = LogSetting()
        self.dataset = DataSetting()
        self.validating_dataset = DataSetting(name='Validating Dataset')
