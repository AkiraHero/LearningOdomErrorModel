#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/03
# @Author  : Akira Ju
# @Email   : jukaka@pku.edu.cn
# @File    : TrainingManager.py

"""
Manager for training error mapping model for odometry-like localization.
"""

from Configuration.TrainingConfiguration import TrainingConfiguration, ModelSetting
import torch
import torch.multiprocessing
import os
import datetime
import torch.utils.data as torch_data
import torch.nn as nn
import Configuration.SettingInstances as si
from Training.errorMappingModel import *
from Dataset.DatasetLoader.generateLocalizationDataSet import stack_data_sample, change_data_device
from Test.TestManager import with_no_grad


class TrainingManager:
    """
    To manage the training process including data feeding, logging, parameter loading and frame control.
    """
    start_time = None
    learner = None
    dataset = None
    data_loader = None
    configuration = None
    device = None

    def __init__(self, config_file=None):
        """
        Initialize and load training settings.
        :param config_file:
        """
        # todo: variable teasing && machine info
        self.config_file = config_file
        self.__get_machine_info__()
        self.training_status = {'epoch': 0, 'step': 0, 'cnt': 0}
        self.machine_info = {}
        self.epoch_num = 0
        self.train_log_dir = None
        self.load_para = None
        self.max_step_num = 0
        self.log_time_interval = 0
        self.retrieve_history = False
        self.validating = False
        self.validating_interval = 1  # epoch
        self.validating_data_loader = None
        self.validating_losslist = []
        self.__set_configuration__(config_file)
        self.gpu_num = torch.cuda.device_count()

    def set_learner(self, learner):
        """
        Set a learner, and transit necessary configurations to it.
        :param learner: Learner instance.
        """
        self.learner = learner
        self.learner.get_setting(self.configuration.training)
        model_instance, other_para2opt = self._ini_model()
        self.learner.set_deep_model(model_instance, other_paras=other_para2opt)
        if self.device == 'cpu':
            self.learner.set_device(torch.device('cpu'))
        else:
            self.learner.set_device(torch.device('cuda'))

    def set_data(self, dataset, validating_dataset=None):
        """
        Set dataset.
        :param dataset:
        """
        self.dataset = dataset
        assert isinstance(dataset, torch_data.Dataset)
        setting = self.configuration.training
        assert isinstance(setting, si.TrainingSetting)
        #for multi-worker
        torch.multiprocessing.set_sharing_strategy('file_system')
        self.data_loader = torch_data.DataLoader(dataset=dataset,
                                                 batch_size=setting.batch_size,
                                                 num_workers=setting.dataloader_worker,
                                                 shuffle=setting.shuffle,
                                                 drop_last=True,
                                                 pin_memory=True
                                                 )
        if self.configuration.training.validating:
            assert validating_dataset is not None
            self.validating_data_loader = torch_data.DataLoader(validating_dataset, batch_size=setting.batch_size,
                                                 num_workers=0, shuffle=False, drop_last=True)

    def _ini_model(self):
        """
        Initialize model class(network module).
        :return:
        """
        assert isinstance(self.configuration.model, ModelSetting)
        other_para2opt = None
        if self.device == 'cuda':
            model = eval(self.configuration.model.classname + '().cuda()')
            if self.configuration.training.learn_loss_weight:
                sigma1 = torch.tensor(-2.0).cuda()
                sigma2 = torch.tensor(0.0).cuda()
                sigma1.requires_grad = True
                sigma2.requires_grad = True
                other_para2opt = {'sigma1': sigma1, 'sigma2': sigma2}
            # enable multi-GPU
            if self.gpu_num > 1:
                model = nn.DataParallel(model)
        else:
            model = eval(self.configuration.model.classname + '()')
            if self.configuration.training.learn_loss_weight:
                sigma1 = torch.tensor(0.0, requires_grad=True)
                sigma2 = torch.tensor(0.0, requires_grad=True)
                other_para2opt = {'sigma1': sigma1, 'sigma2': sigma2}
        return model, other_para2opt

    def __get_machine_info__(self):
        """
        Get starting time or ther machine information.
        """
        # Get training time
        self.start_time = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
        self.last_params_saving_time = datetime.datetime.now().timestamp()

    def __set_configuration__(self, file):
        """
        Set configuration.
        :param file:
        """
        self.configuration = TrainingConfiguration(file)
        self.epoch_num = self.configuration.training.epoch
        self.train_log_dir = os.path.join(self.configuration.logger.dir, self.configuration.logger.prefix)
        self.max_step_num = self.configuration.training.max_step_num
        self.log_time_interval = self.configuration.logger.log_time_interval
        self.validating = self.configuration.training.validating
        self.validating_interval = self.configuration.training.validating_interval

        assert self.configuration.training.load_para ^ self.configuration.training.retrieve_train or \
               (self.configuration.training.load_para == False and False == self.configuration.training.retrieve_train)

        self.retrieve_history = self.configuration.training.retrieve_train
        if self.configuration.training.load_para:
            self.load_para = self.configuration.training.para_file

        if not self.configuration.training.device_id == 'cpu':
            os.environ["CUDA_VISIBLE_DEVICES"] = self.configuration.training.device_id
            self.device = 'cuda'
        else:
            self.device = 'cpu'
        if not os.path.exists(self.train_log_dir):
            os.makedirs(self.train_log_dir)
        self.configuration.write_config(os.path.join(self.train_log_dir, 'config-' + self.start_time + '.ini'))
        self.txt_log = open(os.path.join(self.train_log_dir, 'loss-' + self.start_time + '.txt'), 'w')
        if self.validating:
            self.txt_log_validating = open(os.path.join(self.train_log_dir, 'validation-' + self.start_time + '.txt'), 'w')

    def save_parameter(self, model_instance):
        """
        Save parameters at checkpoints.
        :param model_instance:
        """
        epoch = self.training_status['epoch']
        step = self.training_status['step']
        cnt = self.training_status['cnt']
        log_path = os.path.join(self.train_log_dir, 'model_params_epoch{}_step{}_cnt{}-{}.pkl'.
                                format(epoch, step, cnt, self.start_time))
        torch.save(model_instance.state_dict(), log_path)
        self.last_params_saving_time = datetime.datetime.now().timestamp()

    def load_training_history(self):
        """
        Load parameters from checkpoints.
        :return:
        """
        assert isinstance(self.configuration.training, si.TrainingSetting)
        global lastpkl

        epochnow = -1
        stepnow = -1
        if self.configuration.training.retrieve_train:
            assert os.path.exists(self.train_log_dir)
            pkllist = os.listdir(self.train_log_dir)
            for pkl in pkllist:
                if -1 == pkl.find('params'):
                    continue
                epoch = int(pkl.split('_')[2].lstrip('epoch'))
                if epoch > epochnow:
                    epochnow = epoch
            for pkl in pkllist:
                if -1 == pkl.find('params'):
                    continue
                epoch = int(pkl.split('_')[2].lstrip('epoch'))
                step = int(pkl.split('_')[3].lstrip('step'))
                if epoch == epochnow and step > stepnow:
                    stepnow = step
                    lastpkl = os.path.join(self.train_log_dir, pkl)
        return epochnow, stepnow, lastpkl

    def train(self):
        """
        Start training.
        """
        self.print_settings()
        assert self.learner and self.data_loader
        # Check History
        if self.retrieve_history:
            epochnow, stepnow, lastpkl = self.load_training_history()
            if self.device == 'cpu':
                paradict = torch.load(lastpkl, map_location=torch.device('cpu'))
            else:
                paradict = torch.load(lastpkl, map_location=torch.device('cuda'))

            self.learner.deep_model.load_state_dict(paradict)
            print("Retrieve Para:", lastpkl)
        else:
            epochnow = -1
            stepnow = -1
        if self.load_para:
            if self.device == 'cpu':
                paradict = torch.load(self.load_para, map_location=torch.device('cpu'))
            else:
                paradict = torch.load(self.load_para, map_location=torch.device('cuda'))

            self.learner.deep_model.load_state_dict(paradict)
            print("Load Para:", self.load_para)
        cnt = 0
        step_num = len(self.dataset) // self.learner.batch_size
        for epoch in range(self.epoch_num):
            self.training_status['step'] = 0
            self.training_status['epoch'] = epoch
            if epoch < epochnow:
                continue
            if epoch % self.configuration.logger.log_epoch_interval == 0:
                self.save_parameter(self.learner.deep_model)
            validatenow = self.validating and epoch >0 and (epoch % self.validating_interval == 0)
            if validatenow:
                print("=======Start Validation=======")
                self.do_validation()
            for step, data in enumerate(self.data_loader):
                if epoch == epochnow and step <= stepnow:
                    continue
                stack_data_sample(data)
                change_data_device(data, self.device)
                self.training_status['step'] = step
                if self.max_step_num != 0 and cnt > self.max_step_num - 1:
                    break
                cnt += 1
                self.training_status['cnt'] = cnt
                self.learner.step_back_propagation(data)
                print_info = "Epc:{}/{}, Stp:{}/{}, Cnt:{}/{}, BSize:{}, Tstep:{} -- ".format(
                    epoch, self.epoch_num, step, step_num, cnt, self.max_step_num, self.learner.batch_size,
                    self.learner.time_step_num) + self.learner.get_step_info()
                print(print_info)
                train_info = self.learner.get_training_information
                # todo: using logger class, 1.get items in the dict 2.print them into file (add a header)
                self.txt_log.write("{},{},{:.5f},{:.5f},{:.5f}\n".
                      format(epoch, step, train_info['loss'], train_info['error_yaw'], train_info['finalpt_se']))
                self.txt_log.flush()
                timestamp = datetime.datetime.now().timestamp()
                if timestamp - self.last_params_saving_time > self.log_time_interval * 60 or cnt % 1000 == 0:
                    self.save_parameter(self.learner.deep_model)
            # TODO: add try-exception to cope with interuptions
        self.save_parameter(self.learner.deep_model)

    # TODO: make it work.
    @with_no_grad()
    def do_validation(self):
        """
        Perform validation.
        :return:
        """
        step_num = len(self.validating_data_loader)
        print("Total Validation steps", step_num)
        if step_num==0:
            print("Validation Failed: dataset is too small. Cancel validation!")
            self.validating = False
            return
        sum_loss = 0
        sum_yawerr = 0
        sum_SEerr = 0
        cnt = 0
        for step, data in enumerate(self.validating_data_loader):
            stack_data_sample(data)
            change_data_device(data, self.device)
            self.learner.step_back_propagation(data, doBP=False)
            train_info = self.learner.get_training_information
            print_info = "Validation::Stp:{}/{}, loss={:.2f}, yaw_err={:.6f}, PoseSE={:.6f}". \
                format(step,
                       step_num,
                       train_info['loss'],
                       train_info['error_yaw'],
                       train_info['finalpt_se'])
            print(print_info)
            cnt += 1
            sum_loss += train_info['loss']
            sum_yawerr += train_info['error_yaw']
            sum_SEerr += train_info['finalpt_se']

        self.validating_losslist.append(sum_loss)
        print("=======Validation End with Loss:{}=======".format(sum_loss))
        self.txt_log_validating.write("Before the beginning of epoch:{},sum_valiLoss:{},sum_yawError:{},sum_SEError:{}\n".
                                      format(self.training_status['epoch'],
                                             sum_loss,
                                             sum_yawerr,
                                             sum_SEerr))
        self.txt_log_validating.flush()

    def print_settings(self):
        """
        Print training settings.
        """
        print("=========Configuration Review=========")
        self.configuration.display_all()
        print("=========Training begin=========")







