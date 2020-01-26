#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/03
# @Author  : Akira Ju
# @Email   : jukaka@pku.edu.cn
# @File    : TestManager.py

"""
Testing error mapping model for odometry-like localization.
"""

from Configuration.TestConfiguration  import TestConfiguration
import torch
import torch.multiprocessing
import os
import datetime
import torch.utils.data as torch_data
import Configuration.SettingInstances as si
from Training.errorMappingModel import *
from Training.ErrorModelLearner import ErrorModelLearner, RelativePositionFuser, cal_pose_error2d
from Dataset.DatasetLoader.generateLocalizationDataSet import stack_data_sample, change_data_device
from functools import wraps
from Utils.fusionUtils import uniform_trans_matrix2d, get_pose2d
import numpy as np
from collections import OrderedDict
from Utils.rigidTransform import fast_matmul


def with_no_grad():
    """
    Define a with_no_grad function for decorator usages

    :return: wrapped function
    """
    def wrap(func):
        @wraps(func)
        def func_with_no_grad(*args, **kwargs):
            with torch.no_grad():
                return func(*args, **kwargs)
        return func_with_no_grad
    return wrap


class TestManager:
    """
    Test Manager.
    """
    learner = None
    dataset = None
    data_loader = None
    configuration = None
    device = None

    def __init__(self, config_file=None):
        # todo: variable teasing && machine info
        self.__get_machine_info__()
        self.machine_info = {}
        self.test_log_dir = None
        self.__set_configuration__(config_file)
        self.gpu_num = torch.cuda.device_count()
        self.data_sample_num = 0
        self.model_parallel = False

    @with_no_grad()
    def set_learner(self, learner):
        """
        Set a learner, and transit necessary configurations to it.
        :param learner: Learner instance.
        """
        self.learner = learner
        self.learner.dataset_len = self.data_sample_num
        self.learner.get_setting(self.configuration.testing)
        model_instance = self._ini_model()
        self.learner.set_deep_model(model_instance)
        if self.device == 'cpu':
            self.learner.set_device(torch.device('cpu'))
        else:
            self.learner.set_device(torch.device('cuda'))
        try:
            self.load_model_params(self.configuration.testing.para_file)
        except RuntimeError:
            if not self.model_parallel:
                model_instance = nn.DataParallel(model_instance)
                self.model_parallel = True
                self.learner.set_deep_model(model_instance)
                self.load_model_params(self.configuration.testing.para_file)
            else:
                self.model_parallel = True
                self.learner.set_deep_model(model_instance)
                self.load_model_params(self.configuration.testing.para_file, addmodule=True)

    @with_no_grad()
    def set_data(self, dataset):
        """
        Set dataset.
        :param dataset:
        """
        self.dataset = dataset
        self.data_sample_num = len(dataset)
        assert isinstance(dataset, torch_data.Dataset)
        #for multi-worker
        torch.multiprocessing.set_sharing_strategy('file_system')
        self.data_loader = torch_data.DataLoader(dataset=dataset, batch_size=self.configuration.testing.batch_size,
                                                 num_workers=self.configuration.testing.dataloader_worker,
                                                 shuffle=False, drop_last=False, pin_memory=True)

    @with_no_grad()
    def _ini_model(self):
        """
        Initialize model class(network module).
        :return:
        """
        assert isinstance(self.configuration.model, si.ModelSetting)
        if self.device == 'cuda':
            model = eval(self.configuration.model.classname + '().cuda()')
            # enable multi-GPU
            if self.gpu_num > 1:
                model = nn.DataParallel(model)
                self.model_parallel = True
            else:
                self.model_parallel = False
        else:
            model = eval(self.configuration.model.classname + '()')
        return model

    def __get_machine_info__(self):
        """
        Get starting time or ther machine information.
        """
        # Get training time
        self.start_time = datetime.datetime.now().strftime('%Y%m%d%H%M%S')

    def __set_configuration__(self, file):
        """
        Set configuration.
        :param file:
        """
        self.configuration = TestConfiguration(file)
        self.test_log_dir = os.path.join(self.configuration.logger.log_dir, self.start_time)

        if not self.configuration.testing.device_id == 'cpu':
            os.environ["CUDA_VISIBLE_DEVICES"] = self.configuration.testing.device_id
            self.device = 'cuda'
        else:
            self.device = 'cpu'
        if not os.path.exists(self.test_log_dir):
            os.makedirs(self.test_log_dir)
        self.configuration.write_config(os.path.join(self.test_log_dir, 'config-' + self.start_time + '.ini'))

    @with_no_grad()
    def load_model_params(self, pklfile, addmodule=False, delmodule=False):
        """
        Load model parameters in *.pkl training log files.
        :param pklfile:  parameters dictionary file.
        :param addmodule: add prefix "module." for items from paradict. For load log from training process using single card.
        :param delmodule: delete prefix "module." for items from paradict. For load log from training process using multiple cards.
        """
        if self.device == 'cpu':
            paradict = torch.load(pklfile, map_location=torch.device('cpu'))
        else:
            paradict = torch.load(pklfile, map_location=torch.device('cuda'))
        if not addmodule == delmodule == False:
            assert addmodule ^ delmodule
        if addmodule:
            paradict_n = OrderedDict()
            for key in paradict.keys():
                assert 'module' not in key
                key_n = 'module.' + key
                paradict_n[key_n] = paradict[key]
            paradict = paradict_n
        if delmodule:
            paradict_n = OrderedDict()
            for key in paradict.keys():
                assert 'module' == key.split('.')[0]
                key_n = '.'.join(key.split('.')[1:])
                paradict_n[key_n] = paradict[key]
            paradict = paradict_n
        self.learner.deep_model.load_state_dict(paradict)
        self.learner.deep_model.eval()

    @with_no_grad()
    def get_model_input(self, data):
        """
        Preprocess the input data for network.
        :param data:
        :return:
        """
        scene_image = torch.unsqueeze(data['sceneImg'], 2)
        return scene_image

    def save_output(self, batchcnt, outputinfo, format='txt', prefix='NetPrediction'):
        """
        Save info of current mini-batch to log files.
        :param batchcnt: batch index.
        :param outputinfo: string.
        :param format: format suffix of log file.
        :param prefix: prefix of log file.
        :return: log file path.
        """
        batchsize = outputinfo.shape[0]
        file_dir = self.test_log_dir
        filep = os.path.join(file_dir, prefix + "-{}.{}".format(self.start_time, format))
        with open(filep, 'a') as f:
            for j in range(batchsize):
                inxstr = "{}".format(batchcnt * self.configuration.testing.batch_size + j + 1)
                s = inxstr + ',' + ','.join(["{:.6E}".format(i) for i in outputinfo[j].cpu().detach().numpy().reshape((-1,))]) + '\n'
                f.write(s)
        return filep

    @with_no_grad()
    def model_prediction(self):
        """
        Testing Mode: get all frames of the raw output of network model using test dataset as input
        :return: result file.
        """
        self.print_settings()
        # force the timestep = 1
        self.learner.time_step_num = 1
        step_num = len(self.data_loader)
        cnt = 0
        for step, data in enumerate(self.data_loader):
            stack_data_sample(data)
            change_data_device(data, self.device)
            inputdata = self.get_model_input(data)
            outputinfo = self.learner.net_step_output(inputdata)
            # todo: get the size of output, save as txt
            fileout = self.save_output(cnt, outputinfo, format='csv')
            print_info = "Model Prediction== Step:{}/{}, batch_size = {}".format(
                step, step_num, self.learner.batch_size)
            print(print_info)
            cnt += 1
        return fileout

    @with_no_grad()
    def visualize_covariance_sample(self):
        """
        Testing Mode: visualize all frames of the covariance output of network model using test dataset as input, and
                        write to a video.

        """
        self.print_settings()
        # import necessary for plot
        import numpy as np
        import matplotlib
        matplotlib.use('Agg')
        from Utils.drawCovSamples import plot_cov2d
        import matplotlib.pyplot as plt
        plt.ioff()
        import sys
        for i in sys.path:
            if 'ros' in i:
                sys.path.remove(i)
        import cv2 as cv

        # force the timestep = 1
        self.learner.time_step_num = 1
        cnt = 0
        use_exp_for_LLT = self.configuration.testing.use_exp_for_LLT
        file_dir = self.test_log_dir
        filep = os.path.join(file_dir, 'CovSample_video' + "-{}.avi".format(self.start_time))

        # initialize the video writer
        videosize = (900, 900)
        videowriter = cv.VideoWriter(filep, cv.VideoWriter_fourcc('M', 'J', 'P', 'G'), 10, videosize)
        # begin testing
        for step, data in enumerate(self.data_loader):
            stack_data_sample(data)
            change_data_device(data, self.device)
            inputdata = self.get_model_input(data)
            outputinfo = self.learner.net_step_output(inputdata)
            information_matrix = ErrorModelLearner.get_information_matrix(outputinfo[:, 0, :], expfordiag=use_exp_for_LLT)
            for i in range(outputinfo.shape[0]):
                fig = plt.figure(figsize=(9, 9))
                title = "frameInx:{}".format(i)
                plt.subplot(2, 3, 1)
                scene = data['sceneImg'][i].cpu().numpy().astype(int).squeeze(0)
                plt.imshow(scene)
                if 'lotradcov-comp' in data.keys():
                    plt.subplot(2, 3, 2)
                    plot_cov2d(data['lotradcov-comp'][i][:, 1:].reshape(3,3).cpu().numpy()[:2, :2], title="Trad-Comp:" + title)

                if 'lotradcov' in data.keys():
                    plt.subplot(2, 3, 3)
                    plot_cov2d(data['lotradcov'][i][:, 1:].reshape(3,3).cpu().numpy()[:2, :2], title="Trad:" + title)

                plt.subplot(2, 3, 4)
                cov_learned = np.linalg.inv(information_matrix[i].cpu().numpy())[:2, :2]

                try:
                    plot_cov2d(cov_learned, title="Learned" + title)
                except ValueError:
                    print("Fail to draw this frame== batch:{}, inx:{}.".format(step, i))

                fig.canvas.draw()
                fig_img_data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
                img = fig_img_data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
                img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
                videowriter.write(img)

                cnt += 1
                print("Write frm:", cnt)
                plt.close()

        videowriter.release()

    @with_no_grad()
    def find_best_covscale(self, trad_cov_scale_list):
        """
        Testing Mode: find a best covariance scale for traditional method on test dataset

        :param trad_cov_scale_list: list of scale candidates(float).
        """
        self.print_settings()
        '''
            get setting from configuration
        '''
        batchsize = self.configuration.testing.batch_size
        time_step_num = self.configuration.testing.time_step_num
        step_num = len(self.data_loader)
        result_list = []
        fuse_scale = self.configuration.testing.fuse_scale

        for trad_cov_scale in trad_cov_scale_list:
            print("*************trad_cov_scale={}****************".format(trad_cov_scale))

            '''
                Set fusers
            '''
            fuser_trad = RelativePositionFuser()
            fuser_trad.device = self.device
            fuser_trad.set_batch_size(batchsize)
            fuser_trad.initial_information_matrix = torch.diag(torch.tensor([100000., 100000., 100000.],
                                                                            device=self.device))
            '''
                save statistics for every batch step
            '''
            keys = ['fusion_trad', 'gnd_truth']

            # relative position of every time step
            relative_position_dict = {key: 0 for key in keys}
            relative_unim_dict = {key: torch.eye(3).to(self.device).repeat(time_step_num, batchsize, 1, 1) for key in keys}

            # save accumulated (global, from 0 position) positions from <relative_position_dict> for <batch_size> frames
            position_dict = {key: torch.tensor([[0., 0., 0.]], device=self.device).t().repeat(batchsize, 1, 1) for key in
                             keys}
            position_unim_dict = {key: torch.eye(3).to(self.device).repeat(batchsize, 1, 1) for key in keys}

            # error of final pose of a trajectory
            poseerror = {key: 0 for key in keys[:-1]}

            # save error for <batch_size> frames according to <poseerror>
            error_yaw = {key: torch.tensor([0.]).repeat(batchsize, 1) for key in keys[:-1]}
            finalpt_se = {key: torch.tensor([0.]).repeat(batchsize, 1) for key in keys[:-1]}

            error_yaw_sum = 0
            finalpt_se_sum = 0
            frm = 0
            '''
                begin test
            '''
            for step, data in enumerate(self.data_loader):
                stack_data_sample(data)
                change_data_device(data, self.device)
                real_batch_size = data['odomresult'].shape[0]  # for incomplete batch

                # Initialize the fuser
                fuser_trad.reinit()
                for time_step in range(time_step_num):
                    '''Load data
                        data format: data[a][b][c]
                            a: string, data section name
                            b: batch index
                            c: time step
                    '''
                    # Get the ground truth of this relative movement
                    truth_result = data['gtresult'][:, time_step, 1:].unsqueeze(-1)
                    relative_position_dict['gnd_truth'] = truth_result

                    # Get the relative positioning result from the method we are interested in,
                    # which can be visual odometry or laser odometry
                    odom_result, odom_covariance_velocity_model = ErrorModelLearner.get_odominfo(data, time_step)
                    method_result = data['loresult'][:, time_step, 1:].unsqueeze(-1)

                    # scale the trad_information_matrix to get best result for traditional fusion methods
                    trad_cov_matrix = data['lotradcov'][:, time_step, 1:].reshape(-1, 3, 3) * trad_cov_scale
                    trad_information_matrix = torch.inverse(trad_cov_matrix)
                    relative_position_dict['odom_DR'] = odom_result
                    relative_position_dict['fusion_trad'], _, _ = fuser_trad.step_fusion(odom_result,
                                                                                         odom_covariance_velocity_model,
                                                                                         method_result,
                                                                                         trad_information_matrix,
                                                                                         fuse_scale
                                                                                         )
                    for key in keys:
                        relative_unim_dict[key][time_step][:real_batch_size] = uniform_trans_matrix2d(
                            relative_position_dict[key])

                for key in keys:
                    position_unim_dict[key][:real_batch_size] = fast_matmul(relative_unim_dict[key][:, :real_batch_size])
                    position_dict[key][:real_batch_size] = get_pose2d(position_unim_dict[key][:real_batch_size])

                # calculate error
                for key in keys[:-1]:
                    # data size = batchsize*3*1
                    poseerror[key] = cal_pose_error2d(position_dict[key][:real_batch_size],
                                                      position_dict['gnd_truth'][:real_batch_size])
                    error_yaw[key][:real_batch_size] = poseerror[key][:, 2, :].abs()
                    finalpt_se[key][:real_batch_size] = (poseerror[key][:real_batch_size, :2, :]).transpose(-1, -2) \
                        .matmul(poseerror[key][:real_batch_size, :2, :]).squeeze(-1)

                # save the statistic in appending mode
                error_yaw_sum += error_yaw['fusion_trad'][:real_batch_size].sum()
                finalpt_se_sum += finalpt_se['fusion_trad'][:real_batch_size].sum()
                frm += real_batch_size

                for key in keys[:-1]:
                    self.save_output(step, finalpt_se[key][:real_batch_size],
                                     format='csv', prefix=key + '-finalpt_se' + "-scale" + str(trad_cov_scale))
                    self.save_output(step, error_yaw[key][:real_batch_size],
                                     format='csv', prefix=key + '-error_yaw' + "-scale" + str(trad_cov_scale))
                print_info = "Test_Accuracy== Step:{}/{}, batch_size = {}".format(
                    step, step_num, self.configuration.testing.batch_size)
                print(print_info)
            error_yaw_mean = error_yaw_sum / frm
            error_SE_mean = finalpt_se_sum / frm
            result_list.append([trad_cov_scale, error_yaw_mean.item(), error_SE_mean.item()])
            print("error_yaw_mean: ", error_yaw_mean.item())
            print("error_SE_mean: ", error_SE_mean.item())
        print("=======Summary=======")
        print("trad_cov_scale", "error_yaw_mean", "error_SE_mean")
        for i in result_list:
            print(i[0], i[1], i[2])
        arr = np.array(result_list)
        inx_yaw = arr[:, 1].argmin()
        inx_SE = arr[:, 2].argmin()
        print("When trad_cov_scale={}, we get minimum of error_yaw_mean={}".format(arr[inx_yaw, 0], arr[inx_yaw, 1]))
        print("When trad_cov_scale={}, we get minimum of error_SE_mean={}".format(arr[inx_SE, 0], arr[inx_SE, 2]))


    @with_no_grad()
    def compare_accuracy(self, gen_sampletraj=False, withNetPre=False):
        """
        Testing Mode: Compare positioning accuracy of different methods

        :param gen_sampletraj: save all trajectories from different methods if True
        :param withNetPre: use pre-computed results from model_prediction if True
        """
        self.print_settings()
        '''
            get setting from configuration
        '''
        batchsize = self.configuration.testing.batch_size
        time_step_num = self.configuration.testing.time_step_num
        use_exp_for_LLT = self.configuration.testing.use_exp_for_LLT
        step_num = len(self.data_loader)
        cnt = 0  # frame counter
        trad_cov_scale = self.configuration.testing.trad_cov_scale
        cal_trad_fusion = self.configuration.testing.cal_trad_fusion
        fuse_scale = self.configuration.testing.fuse_scale



        '''
            Set fusers
        '''
        fuser = RelativePositionFuser()
        fuser.device = self.device
        fuser.set_batch_size(batchsize)
        fuser.initial_information_matrix = torch.diag(torch.tensor([100000., 100000., 100000.],
                                                                   device=self.device))
        if cal_trad_fusion:
            fuser_trad = RelativePositionFuser()
            fuser_trad.device = self.device
            fuser_trad.set_batch_size(batchsize)
            fuser_trad.initial_information_matrix = torch.diag(torch.tensor([100000., 100000., 100000.],
                                                                            device=self.device))
        '''
            save statistics for every batch step
        '''
        if cal_trad_fusion:
            keys = ['fusion_learned', 'odom_target', 'odom_DR', 'fusion_trad', 'gnd_truth']
        else:
            keys = ['fusion_learned', 'odom_target', 'odom_DR', 'gnd_truth']

        # relative position of every time step
        relative_position_dict = {key: 0 for key in keys}
        relative_unim_dict = {key: torch.eye(3).to(self.device).repeat(time_step_num, batchsize, 1, 1) for key in keys}

        # save accumulated (global, from 0 position) positions from <relative_position_dict> for <batch_size> frames
        position_dict = {key: torch.tensor([[0., 0., 0.]], device=self.device).t().repeat(batchsize, 1, 1) for key in keys}
        position_unim_dict = {key: torch.eye(3).to(self.device).repeat(batchsize, 1, 1) for key in keys}
        if gen_sampletraj:
            traj_unim_dict = {key: torch.eye(3).to(self.device).repeat(batchsize, time_step_num + 1, 1, 1) for key in keys}
            traj_dict = {key: torch.tensor([[0., 0., 0.]], device=self.device).t().repeat(batchsize, time_step_num + 1, 1, 1) for key in keys}

        # error of final pose of a trajectory
        poseerror = {key: 0 for key in keys[:-1]}

        # save error for <batch_size> frames according to <poseerror>
        error_yaw = {key: torch.tensor([0.]).repeat(batchsize, 1) for key in keys[:-1]}
        finalpt_se = {key: torch.tensor([0.]).repeat(batchsize, 1) for key in keys[:-1]}
        '''
            begin test
        '''
        for step, data in enumerate(self.data_loader):
            stack_data_sample(data)
            change_data_device(data, self.device)
            inputdata = self.get_model_input(data)
            if not withNetPre:
                list_information_descriptor = self.learner.net_step_output(inputdata)
            real_batch_size = data['odomresult'].shape[0]  # for incomplete batch

            # Initialize the fuser
            fuser.reinit()
            if cal_trad_fusion:
                fuser_trad.reinit()
            # print("P1")

            for time_step in range(time_step_num):
                '''Load data
                    data format: data[a][b][c]
                        a: string, data section name
                        b: batch index
                        c: time step
                '''
                # Get the ground truth of this relative movement
                truth_result = data['gtresult'][:, time_step, 1:].unsqueeze(-1)
                relative_position_dict['gnd_truth'] = truth_result

                # Get the relative positioning result from the method we are interested in,
                # which can be visual odometry or laser odometry
                method_result = data['loresult'][:, time_step, 1:].unsqueeze(-1)
                odom_result, odom_covariance_velocity_model = ErrorModelLearner.get_odominfo(data, time_step)

                # scale the trad_information_matrix to get best result for traditional fusion methods
                if cal_trad_fusion:
                    empty_batch_inx = (time_step >= data['seglength']).nonzero()
                    trad_cov_matrix = data['lotradcov'][:, time_step, 1:].reshape(-1, 3, 3) * trad_cov_scale
                    trad_cov_matrix[empty_batch_inx] = torch.eye(3, device=self.device) * 1e-16
                    trad_information_matrix = torch.inverse(trad_cov_matrix)

                relative_position_dict['odom_target'] = method_result
                relative_position_dict['odom_DR'] = odom_result

                # Get the learned information matrix
                if not withNetPre:
                    information_descriptor = list_information_descriptor[:, time_step, :]
                    method_information_matrix = ErrorModelLearner.get_information_matrix(information_descriptor,
                                                                            expfordiag=use_exp_for_LLT)
                else:
                    method_information_matrix = data['modelprediction_infomat'][:, time_step, 1:].reshape(-1, 3, 3)

                # Fusion using Information Filter
                # TODO: warning!!!!!!!!! unstable numerical results!

                '''
                 odom_covariance_velocity_model: using covariance
                '''
                relative_position_dict['fusion_learned'], _, _ = fuser.step_fusion(odom_result,
                                                                                   odom_covariance_velocity_model,
                                                                                   method_result,
                                                                                   method_information_matrix,
                                                                                   fuse_scale
                                                                                   )
                if cal_trad_fusion:
                    relative_position_dict['fusion_trad'], _, _ = fuser_trad.step_fusion(odom_result,
                                                                                         odom_covariance_velocity_model,
                                                                                         method_result,
                                                                                         trad_information_matrix,
                                                                                         fuse_scale
                                                                                         )
                if not fuse_scale:
                    scale = (relative_position_dict['odom_DR'][..., 0, :] ** 2 + relative_position_dict['odom_DR'][..., 1, :] ** 2).sqrt().unsqueeze(-1)
                    relative_position_dict['odom_target'][..., :2, :] = relative_position_dict['odom_target'][..., :2, :] * scale

                for key in keys:
                    relative_unim_dict[key][time_step][:real_batch_size] = uniform_trans_matrix2d(relative_position_dict[key])

            # print("P2")
            for key in keys:
                position_unim_dict[key][:real_batch_size] = fast_matmul(relative_unim_dict[key][:, :real_batch_size])
                position_dict[key][:real_batch_size] = get_pose2d(position_unim_dict[key][:real_batch_size])
            # print("P3")
            if gen_sampletraj:
                for key in keys:
                    traj_unim_dict[key][:real_batch_size, 0] = torch.eye(3).to(self.device).repeat(real_batch_size, 1, 1)
                    for tstep in range(1, time_step_num + 1):
                        traj_unim_dict[key][:real_batch_size, tstep] = traj_unim_dict[key][:real_batch_size, tstep - 1]\
                            .matmul(relative_unim_dict[key][tstep - 1, :real_batch_size])
                        traj_dict[key][:real_batch_size, tstep] = get_pose2d(traj_unim_dict[key][:real_batch_size, tstep])
                    # save traj
                    self.save_output(step, traj_dict[key][:real_batch_size], format='csv', prefix=key + '-batchtraj-timestepnum{}'.format(time_step_num))
            # print("P4")



            # calculate error
            for key in keys[:-1]:
                # data size = batchsize*3*1
                poseerror[key] = cal_pose_error2d(position_dict[key][:real_batch_size],
                                                  position_dict['gnd_truth'][:real_batch_size])
                error_yaw[key][:real_batch_size] = poseerror[key][:, 2, :].abs()
                finalpt_se[key][:real_batch_size] = (poseerror[key][:real_batch_size, :2, :]).transpose(-1, -2)\
                    .matmul(poseerror[key][:real_batch_size, :2, :]).squeeze(-1)

            # save the statistic in appending mode
            for key in keys[:-1]:
                self.save_output(step, finalpt_se[key][:real_batch_size], format='csv', prefix=key + '-finalpt_se')
                self.save_output(step, error_yaw[key][:real_batch_size], format='csv', prefix=key + '-error_yaw')

            print_info = "Test_Accuracy== Step:{}/{}, batch_size = {}".format(
                step, step_num, self.learner.batch_size)
            print(print_info)
            cnt += 1
            # print("P5")

    def print_settings(self):
        """
        print test settings
        """
        print("=========Configuration Review=========")
        self.configuration.display_all()
        print("=========Test begin=========")

    @staticmethod
    def transNetPrediction2Mat(infile, outfile, type):
        """
        Transform the raw model prediction using inverse Cholesky factorization to covariance/information matrix.
        Note: for straight LLT form without any preprocessing.
        :param infile:  results file from model_prediction
        :param outfile: output results file
        :param type: cov/info
        """
        print("We are transform raw NetPrediction to Cov/Info mat, using LLT form!")
        import pandas as pd
        arr = pd.read_csv(infile, header=None).values[:, 1:]
        arr_n = np.zeros(arr.shape)
        arr_n[:, 0] = arr[:, 0]
        arr_n[:, 1] = arr[:, 3]
        arr_n[:, 2] = arr[:, 1]
        arr_n[:, 3] = arr[:, 4]
        arr_n[:, 4] = arr[:, 5]
        arr_n[:, 5] = arr[:, 2]
        ltriarr =  np.zeros([arr.shape[0], 3, 3])
        ltrinx_l, ltrinx_r = np.tril_indices(3)
        ltriarr[:, ltrinx_l, ltrinx_r] = arr_n
        rtriarr = ltriarr.transpose(0, 2, 1)
        Mat = torch.tensor(ltriarr).matmul(torch.tensor(rtriarr))
        if type == 'cov':
            Mat = Mat.inverse()
        else:
            assert type == 'info'
        cnt = 1
        with open(outfile, 'w') as f:
            for mat in Mat:
                vec = mat.reshape(-1,)
                s = ','.join(['{:.0f}'] + ['{:.6E}'] * 9).format(cnt, *vec) + '\n'
                f.write(s)
                cnt += 1




