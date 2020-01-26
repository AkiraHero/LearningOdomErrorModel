#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/03
# @Author  : Akira Ju
# @Email   : jukaka@pku.edu.cn
# @File    : ErrorModelLearner.py

import torch
from Utils.fusionUtils import get_relative_unim_pose2d, get_velocity_model_covariance, \
    predict_next_deltapose2d, inverse_uniform_matrix2d,\
    cal_pose_error2d, unim_pose2d_aggregation,uniform_trans_matrix2d, get_pose2d
from Utils.relativePositionFuser import RelativePositionFuser
from Configuration.SettingInstances import TrainingSetting, TestSetting
from .Learner import Learner
from Utils.rigidTransform import fast_matmul


class ErrorModelLearner(Learner):
    """
    Class ErrorModelLearner: class for training the error mapping model of odometry-like localization.
    """
    def __init__(self):
        super(ErrorModelLearner, self).__init__()
        self.dataset_len = None
        self.time_step_num = None
        self.loss_function = None
        self.fuser = RelativePositionFuser()
        self.training_infomation = {}
        self.loss_info_dict = {}
        # save data of a training sample
        self.sample_data_dict = {}
        self.learn_loss_weight = False
        self.yawloss_weight = 1.0
        self.use_exp_for_LLT = False
        self.add_reverse_traj_loss = False
        self.fuse_scale = False

    def get_setting(self, setting):
        """
        Add setting to this learner
        :param setting:
        """
        if isinstance(setting, TrainingSetting):
            self.batch_size = setting.batch_size
            self.time_step_num = setting.time_step_num
            self.learning_rate = setting.learning_rate
            self.weight_decay = setting.weight_decay
            self.optimizer = setting.optimizer
            self.yawloss_weight = setting.yawloss_weight
            self.use_exp_for_LLT = setting.use_exp_for_LLT
            self.fuser.set_batch_size(self.batch_size)
            self.learn_loss_weight = setting.learn_loss_weight
            self.add_reverse_traj_loss = setting.add_reverse_traj_loss
            self.fuse_scale = setting.fuse_scale
        elif isinstance(setting, TestSetting):
            self.batch_size = setting.batch_size
            self.time_step_num = setting.time_step_num
        else:
            assert True

    def set_loss_function(self, loss_func):
        """
        Set loss function
        :param loss_func: function name.
        """
        self.loss_function = loss_func

    def step_back_propagation(self, data, doBP=True):
        """
        Step operation including [Forward prediction] and optional [Back propagation]
        :param data: input data of this frame
        :param doBP: do BP if True
        """
        relative_unim_model_fusion = torch.eye(3).to(self.device).repeat(self.time_step_num, self.batch_size, 1, 1)
        relative_unim_gndtruth = torch.eye(3).to(self.device).repeat(self.time_step_num, self.batch_size, 1, 1)
        zero_unim = torch.eye(3).to(self.device).repeat(self.batch_size, 1, 1)

        # Set initial information matrix for the fuser
        self.fuser.initial_information_matrix = torch.diag(torch.tensor([100000., 100000., 100000.], device=self.device))

        """Forward prediction
        """
        # 1.Load scene image and get the information descriptor

        #   Make the size of scene_image: (batch_size * time_step_num) * 1 * height * width
        scene_image = torch.unsqueeze(data['sceneImg'], 2)

        #   Get independent elements for the information matrix
        list_information_descriptor = self.net_step_output(scene_image)

        # 2.Load other data and then do fusion using information filter based fuser
        # Initialize the fuser
        self.fuser.reinit()
        for time_step in range(self.time_step_num):
            '''Load data
                data format: data[a][b][c]
                    a: string, data section name
                    b: batch index
                    c: time step
            '''
            # Get the ground truth of this relative movement, shape=batch_size*3*1
            truth_result = data['gtresult'][:, time_step, 1:].unsqueeze(-1)

            # Get the relative positioning result from the method we are interested in,
            # which can be visual odometry or laser odometry, shape=batch_size*3*1
            method_result = data['loresult'][:, time_step, 1:].unsqueeze(-1)
            odom_result, odom_covariance_velocity_model = self.get_odominfo(data, time_step)

            # Get the learned information matrix
            information_descriptor = list_information_descriptor[:, time_step, :]
            method_information_matrix = self.get_information_matrix(information_descriptor, expfordiag=self.use_exp_for_LLT)

            # Fuse method_result[i] odom_result[i] using Information Filter
            # TODO: warning!!!!!!!!! unstable numerical results!

            '''
            @odom_covariance_velocity_model: using covariance
            '''

            fusion_result, _, _ = self.fuser.step_fusion(odom_result,
                                                         odom_covariance_velocity_model,
                                                         method_result,
                                                         method_information_matrix,
                                                         self.fuse_scale
                                                         )

            relative_unim_model_fusion[time_step] = uniform_trans_matrix2d(fusion_result)
            relative_unim_gndtruth[time_step] = uniform_trans_matrix2d(truth_result)

        position_model_fusion_unim = fast_matmul(relative_unim_model_fusion)
        position_model_fusion = get_pose2d(position_model_fusion_unim)
        position_gndtruth_unim = fast_matmul(relative_unim_gndtruth)
        position_gndtruth = get_pose2d(position_gndtruth_unim)
        poseerror = cal_pose_error2d(position_model_fusion, position_gndtruth)
        error_yaw = poseerror[:, 2, :].abs()
        finalpt_se = poseerror[:, :2, :].transpose(-1, -2).matmul(poseerror[:, :2, :]).squeeze(-1)

        if self.add_reverse_traj_loss:
            rela_unim_fusion = get_relative_unim_pose2d(zero_unim, position_model_fusion_unim)
            rela_unim_truth = get_relative_unim_pose2d(zero_unim, position_gndtruth_unim)
            position_model_fusion_re = get_pose2d(self.reverse_matrix(rela_unim_fusion))
            position_gndtruth_re = get_pose2d(self.reverse_matrix(rela_unim_truth))
            poseerror = cal_pose_error2d(position_model_fusion_re, position_gndtruth_re)
            error_yaw = (error_yaw + poseerror[:, 2, :].abs()) / 2.0
            finalpt_se = (finalpt_se + poseerror[:, :2, :].transpose(-1, -2).matmul(poseerror[:, :2, :]).squeeze(-1)) / 2.0

        """Back propagation
        """
        # 1. Calculate loss value
        self.loss_info_dict['error_yaw'] = error_yaw
        self.loss_info_dict['finalpt_se'] = finalpt_se
        loss = self.get_loss(learn_weight=self.learn_loss_weight)

        self.training_infomation['loss'] = loss.item()
        self.training_infomation['error_yaw'] = error_yaw.sum().item()
        self.training_infomation['finalpt_se'] = finalpt_se.sum().item()

        # 2. Do BP
        if doBP:
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def get_loss(self, learn_weight=False):
        """
        Calculate loss.
        :param learn_weight: set True when need to learn the balancing factor between Euclidean distance error and rotation error.
        :return: mean loss value of this mini-batch.
        """
        assert self.yawloss_weight >= 0
        if not learn_weight:
            loss = self.loss_info_dict['error_yaw']**2 * self.yawloss_weight + self.loss_info_dict['finalpt_se']
        else:
            loss = (-self.other_para2opt['sigma1']).exp() * (self.loss_info_dict['error_yaw']**2) \
                   + self.other_para2opt['sigma1'] \
                   + (-self.other_para2opt['sigma2']).exp() * self.loss_info_dict['finalpt_se']\
                   + self.other_para2opt['sigma2']
        loss = loss.mean()
        return loss

    @property
    def get_training_information(self):
        """
        Get training information.
        :return:
        """
        return self.training_infomation

    def get_step_info(self):
        """
        Get information at this frame to display.
        :return:
        """
        if self.other_para2opt is not None:
            lossinfo2print = "loss={:.2f}, yaw_err={:.6f}, PoseSE={:.6f}, s1={:.6f}, s2={:.6f}". \
                format(self.training_infomation['loss'],
                       self.training_infomation['error_yaw'],
                       self.training_infomation['finalpt_se'],
                       self.other_para2opt['sigma1'].item(),
                       self.other_para2opt['sigma2'].item()
                       )
        else:
            lossinfo2print = "loss={:.2f}, yaw_err={:.6f}, PoseSE={:.6f}". \
                format(self.training_infomation['loss'],
                       self.training_infomation['error_yaw'],
                       self.training_infomation['finalpt_se'])
        return lossinfo2print

    def net_step_output(self, scene_image):
        """
        Get a step output of network module.
        :param scene_image:
        :return:
        """
        # self.check_data_device(data)
        # scene_image = torch.unsqueeze(data['voSceneImg'], 2)
        cur_shape = scene_image.shape
        assert (cur_shape[0] == self.batch_size or cur_shape[0] == self.dataset_len % self.batch_size) \
               and (cur_shape[1] == self.time_step_num)
        scene_image = scene_image.reshape(cur_shape[0] * cur_shape[1], cur_shape[2], cur_shape[3], cur_shape[4])
        list_information_descriptor = self.deep_model(scene_image)
        list_information_descriptor = list_information_descriptor.reshape(cur_shape[0], self.time_step_num,
                                                                          list_information_descriptor.shape[1])
        return list_information_descriptor

    @staticmethod
    def get_odominfo(data, time_step):
        """
        Get the corresponding self-odometry result, such as wheel encoder/IMU suite, shape=batch_size*3*1
        :param data:
        :param time_step:
        :return:
        """
        dev = data['odomresult'].device
        batchsize = data['odomresult'].shape[0]
        odom_result = data['odomresult'][:, time_step, 1:].unsqueeze(-1)
        odom_covariance_velocity_model = data['odomtradcov'][:, time_step, 1:].reshape(-1, 3, 3)
        return odom_result, odom_covariance_velocity_model


    @staticmethod
    def get_information_matrix(information_descriptor, expfordiag=False):
        """
        Get information matrix from the raw network output.

        Discussion about Cholesky decomposition: LL^T
                It is not necessary to add positive constraint on L's diagonal entries.
                Prove:
                    L = [e1, e2, e3]
                    LL^T = sum(ei*ei^T)
                    so that for ei, sign has no effect on LL^T
                However, if must make L's diagonal entries positive：
                    tri_matrix[0, 0] = information_descriptor[0].exp()
                    tri_matrix[1, 1] = information_descriptor[1].exp()
                    tri_matrix[2, 2] = information_descriptor[2].exp()
                exp is a good mapping, but the gradient may overflow, so that BN layer is necessary.

                Experiments shows current method performs best.
        :param information_descriptor:
        :param expfordiag:
        :return:
        """
        batchsize = information_descriptor.shape[0]
        tri_matrix = torch.diag(torch.tensor([0., 0., 0.], device=information_descriptor.device)).detach().repeat(batchsize, 1, 1)
        if not expfordiag:
            tri_matrix[:, 0, 0] = information_descriptor[:, 0]
            tri_matrix[:, 1, 1] = information_descriptor[:, 1]
            tri_matrix[:, 2, 2] = information_descriptor[:, 2]
        else:
            tri_matrix[:, 0, 0] = information_descriptor[:, 0].exp()
            tri_matrix[:, 1, 1] = information_descriptor[:, 1].exp()
            tri_matrix[:, 2, 2] = information_descriptor[:, 2].exp()
        tri_matrix[:, 1, 0] = information_descriptor[:, 3]
        tri_matrix[:, 2, 0] = information_descriptor[:, 4]
        tri_matrix[:, 2, 1] = information_descriptor[:, 5]
        method_information_matrix = (tri_matrix.matmul(tri_matrix.transpose(-1, -2)))
        return method_information_matrix

    """
    Some utils only used in this file.
    """
    # to be deleted
    @staticmethod
    def add2pose(local_result, pose):
        batch_size = local_result.shape[0]
        zero_st_position = torch.tensor([0., 0., 0.], device=local_result.device).reshape(3, 1).unsqueeze(0).repeat(batch_size, 1, 1)
        delta_ = predict_next_deltapose2d(local_result, zero_st_position, pose)
        return pose + delta_

    @staticmethod
    def aggregate_pose(local_result, pose):
        unim = unim_pose2d_aggregation(pose, local_result)
        return get_pose2d(unim)

    # m: timestepnum*batchsize*3*3
    @staticmethod
    def mul_matrix(m):
        res = m[0]
        for i in m[1:]:
            res = res.matmul(i)
        return res

    # m: timestepnum*batchsize*3*3
    @staticmethod
    def reverse_matrix(m):
        shape = m.shape
        new_m = m.reshape(-1, 3, 3)
        new_m = inverse_uniform_matrix2d(new_m)
        return new_m.reshape(shape)