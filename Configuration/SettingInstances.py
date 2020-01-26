#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/08
# @Author  : Akira Ju
# @Email   : jukaka@pku.edu.cn
# @File    : SettingInstances.py

"""
Various setting group for machine learning.

Coding Usage:

    # Define a class XXXXSetting for a group of settings.

    Class XXXXSetting(Setting):
        # Construction function, set name for your setting group like "TrainingProcess"
        def __init__(self, config=None, name='TrainingProcess'):
            super(XXXXSetting, self).__init__(name, config)

        # Initialize the variables in setting, any variable should be defined without underlined beginning.
        def _initialize_attrib(self):
            self.epoch = 0
            self.batch_size = 0
            ......

        # Load your setting according to different data type.
        def _load(self, _content):
            self.para_file = _content.get('para_file')

"""

from ._Setting import Setting
import os


class TrainingSetting(Setting):
    """
    Settings for training control.
    """
    def __init__(self, config=None, name='TrainingProcess'):
        super(TrainingSetting, self).__init__(name, config)

    def _initialize_attrib(self):
        self.device_id = 0
        self.learning_rate = 0.0001
        self.epoch = 0
        self.batch_size = 0
        self.load_para = False
        self.para_file = ''
        self.time_step_num = 15
        self.weight_decay = 1e-8
        self.retrieve_train = False
        self.shuffle = True
        self.optimizer = ''
        self.max_step_num = 0
        self.validating = False
        self.validating_interval = 1
        self.yawloss_weight = 100.0
        self.use_exp_for_LLT = False
        self.dataloader_worker = 4
        self.learn_loss_weight = False
        self.add_reverse_traj_loss = False
        self.fuse_scale = True

    def _load(self, _content):
        # string
        self.para_file = _content.get('para_file')
        self.optimizer = _content.get('optimizer')
        self.device_id = _content.get('device_id')
        # bool
        self.load_para = _content.getboolean('load_para')
        self.retrieve_train = _content.getboolean('retrieve_train')
        self.shuffle = _content.getboolean('shuffle')
        self.validating = _content.getboolean('validating')
        self.use_exp_for_LLT = _content.getboolean('use_exp_for_LLT')
        self.learn_loss_weight = _content.getboolean('learn_loss_weight')
        self.add_reverse_traj_loss = _content.getboolean('add_reverse_traj_loss')
        self.fuse_scale = _content.getboolean('fuse_scale')
        # int
        self.batch_size = _content.getint('batch_size')
        self.epoch = _content.getint('epoch')
        self.time_step_num = _content.getint('time_step_num')
        self.max_step_num = _content.getint('max_step_num')
        self.dataloader_worker = _content.getint('dataloader_worker')
        self.validating_interval = _content.getint('validating_interval')
        # float
        self.learning_rate = _content.getfloat('learning_rate')
        self.weight_decay = _content.getfloat('weight_decay')
        self.yawloss_weight = _content.getfloat('yawloss_weight')


class DataSetting(Setting):
    """
    Settings for dataset addressing.
    """
    def __init__(self, config=None, name='DataSet'):
        super(DataSetting, self).__init__(name, config)

    def _initialize_attrib(self):
        self.dataset_name = ''
        self.branch = ''
        self.default_home_path = ''
        self.image_size = '50_50'
        self.st_frame = 0
        self.end_frame = None
        self.algo = ''
        self.use_dataseglist = False
        self.segfile_suffix = "default"

    def _load(self, _content):
        self.dataset_name = _content.get('dataset_name')
        self.branch = _content.get('branch')
        self.default_home_path = _content.get('default_home_path')
        self.image_size = _content.get('image_size')
        self.st_frame = _content.getint('st_frame')
        self.end_frame = _content.get('end_frame')
        self.algo = _content.get('algo')
        self.use_dataseglist = _content.getboolean('use_dataseglist')
        if self.use_dataseglist:
            self.segfile_suffix = _content.get('segfile_suffix')
        if self.end_frame == 'None':
            self.end_frame = None
        else:
            self.end_frame = _content.getint('end_frame')


class LogSetting(Setting):
    """
    Settings for writing log (in training).
    """
    def __init__(self, config=None, name='LogSetting'):
        super(LogSetting, self).__init__(name, config)

    def _initialize_attrib(self):
        self.prefix = ''
        self.dir = ''
        self.log_time_interval = '' # min
        self.log_epoch_interval = 1

    def _load(self, _content):
        self.prefix = _content.get('prefix')
        self.dir = _content.get('dir')
        self.log_time_interval = _content.getfloat('log_time_interval')
        self.log_epoch_interval = _content.getint('log_epoch_interval')


class ModelSetting(Setting):
    """
    Settings for model class selection.
    """
    def __init__(self, config=None, name='ModelSetting'):
        super(ModelSetting, self).__init__(name, config)

    def _initialize_attrib(self):
        self.classname = ''

    def _load(self, _content):
        self.classname = _content.get('classname')


class TestSetting(Setting):
    """
    Settings for testing control.
    """
    def __init__(self, config=None, name='TestSetting'):
        super(TestSetting, self).__init__(name, config)

    def _initialize_attrib(self):
        self.para_file = ''
        self.device_id = 0
        self.batch_size = 1
        self.use_exp_for_LLT = False
        self.time_step_num = 0
        self.dataloader_worker = 4
        self.trad_cov_scale = 1.0
        self.cal_trad_fusion = True
        self.fuse_scale = True

    def _load(self, _content):
        self.para_file = _content.get('para_file')
        self.device_id = _content.get('device_id')
        self.batch_size = _content.getint('batch_size')
        self.time_step_num = _content.getint('time_step_num')
        self.use_exp_for_LLT = _content.getboolean('use_exp_for_LLT')
        self.dataloader_worker = _content.getint('dataloader_worker')
        self.trad_cov_scale = _content.getfloat('trad_cov_scale')
        self.cal_trad_fusion = _content.getboolean('cal_trad_fusion')
        self.fuse_scale = _content.getboolean('fuse_scale')

        # # Get necessary information from training configuration file
        # from .TrainingConfiguration import TrainingConfiguration
        # training_config = TrainingConfiguration(self.training_configuration)
        # self.model_name = training_config.model.classname
        # self.training_log_dir = training_config.logger.dir
        # assert self.para_file in os.listdir(self.training_log_dir)


class TestLogSetting(Setting):
    """
    Settings for writing log in testing.
    """
    def __init__(self, config=None, name='TestLogSetting'):
        super(TestLogSetting, self).__init__(name, config)

    def _initialize_attrib(self):
        self.log_dir = ''
        pass

    def _load(self, _content):
        self.log_dir = _content.get('log_dir')
        pass

