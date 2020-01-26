#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/03
# @Author  : Akira Ju
# @Email   : jukaka@pku.edu.cn
# @File    : Learner.py

import torch


class Learner:
    """
    Class Learner: base class for training an machine learning model.
    """
    def __init__(self):
        self.batch_size = None
        self.deep_model = None # modules
        self.other_para2opt = None # other parameters need to be optimized by BP. type: dict.
        self.optimizer = None
        self.params2optimize = None
        self.learning_rate = None
        self.weight_decay = None
        self.device = torch.device('cpu')

    def set_optimizer(self, optimizer, learning_rate, weight_decay=None):
        if optimizer == 'adadelta':
            self.optimizer = torch.optim.Adadelta(self.params2optimize, lr=learning_rate, weight_decay=weight_decay)
        elif optimizer == 'rmsprop':
            self.optimizer = torch.optim.RMSprop(self.params2optimize, lr=learning_rate, weight_decay=weight_decay, momentum=0.4,alpha=0.9)
        elif optimizer == 'adam':
            self.optimizer = torch.optim.Adam(self.params2optimize, lr=learning_rate, weight_decay=weight_decay)
        elif optimizer == 'adagrad':
            self.optimizer = torch.optim.Adagrad(self.params2optimize, lr=learning_rate, weight_decay=weight_decay)

    def set_device(self, device):
        self.device = device
        self.fuser.device = device

    def check_data_device(self, data):
        for i in data.items():
            data[i[0]] = i[1].to(self.device)

    def set_deep_model(self, net, trainflag=True, other_paras=None):
        self.deep_model = net
        self.other_para2opt = other_paras
        if trainflag:
            self.params2optimize = [{'params': net.parameters()}]
            if other_paras is not None:
                for k, v in other_paras.items():
                    self.params2optimize = self.params2optimize + [{'params': v, 'lr':0.1}]
            self.set_optimizer(self.optimizer, self.learning_rate, weight_decay=self.weight_decay)

    def get_setting(self, setting):
        raise NotImplementedError

    def step_back_propagation(self, data):
        raise NotImplementedError

    # Training information such as loss...

        # Dict
    def get_training_information(self):
        raise NotImplementedError

        # String
    def get_step_info(self):
        raise NotImplementedError
