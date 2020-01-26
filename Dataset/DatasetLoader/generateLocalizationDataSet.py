#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/08
# @Author  : Akira Ju
# @Email   : jukaka@pku.edu.cn
# @File    : generateLocalizationDataSet.py

"""
Control different dataset input for different training/testing tasks.
"""


from .seqDataSet import SeqDataset
from .ImageTensorBinLoader import ImageTensorBinLoader
import os
import torch


def config_loc_dataset(config, mode, **othersection):
    """
    Control different dataset input for different training/testing tasks.
    :param config: Configuration instance
    :param mode:  experimental mode
    :param othersection: other options
    :return:
    """
    # sharing data location
    datasetting = config.dataset
    workpath = datasetting.default_home_path
    dataset_name = datasetting.dataset_name
    branch = datasetting.branch

    imgdir = os.path.join(workpath, dataset_name, branch, datasetting.algo, "SceneInfo")
    laserodomfile = os.path.join(workpath, dataset_name, branch, datasetting.algo, 'relPose-targetodom.csv')
    lasertradcovfile = os.path.join(workpath, dataset_name, branch, datasetting.algo, 'tradcovmatrix-targetodom.csv')
    odomlocal = os.path.join(workpath, dataset_name, branch, datasetting.algo, 'relPose-EncoderDR.csv')
    odomlocaltradcovfile = os.path.join(workpath, dataset_name, branch, datasetting.algo, 'tradcovmatrix-EncoderDR.csv')
    gpsgt = os.path.join(workpath, dataset_name, branch, datasetting.algo, 'relPose-gps-gt.csv')
    seglistfile = None
    if datasetting.use_dataseglist:
        if datasetting.segfile_suffix == 'default':
            segfilename = 'relPose-gps-gt.csv.seg'
        else:
            segfilename = 'relPose-gps-gt.csv.seg' + datasetting.segfile_suffix
        seglistfile = os.path.join(workpath, dataset_name, branch, datasetting.algo, segfilename)


    #  fill dataset according to different mode and set time step
    arglist = {}
    if mode == 'train':
        time_step_num = config.training.time_step_num
        arglist['odomfile'] = odomlocal
        arglist['odomtradcovfile'] = odomlocaltradcovfile
        arglist['laserodomfile'] = laserodomfile
        arglist['sceneImgDir'] = imgdir
        arglist['frmset'] = (datasetting.st_frame, datasetting.end_frame)
        arglist['seglistfile'] = seglistfile
        arglist['usingseglist'] = datasetting.use_dataseglist
        if config.training.validating:
            datasetting_v = config.validating_dataset
            workpath_v = datasetting_v.default_home_path
            dataset_name_v = datasetting_v.dataset_name
            branch_v = datasetting_v.branch
            imgdir_v = os.path.join(workpath_v, dataset_name_v, branch_v, datasetting_v.algo, "SceneInfo")
            laserodomfile_v = os.path.join(workpath_v, dataset_name_v, branch_v, datasetting_v.algo, 'relPose-targetodom.csv')
            odomlocal_v = os.path.join(workpath_v, dataset_name_v, branch_v, datasetting_v.algo, 'relPose-EncoderDR.csv')
            odomlocaltradcovfile_v = os.path.join(workpath_v, dataset_name_v, branch_v, datasetting_v.algo,
                                                'tradcovmatrix-EncoderDR.csv')
            gpsgt_v = os.path.join(workpath_v, dataset_name_v, branch_v, datasetting_v.algo, 'relPose-gps-gt.csv')
            seglistfile_v = None
            if datasetting_v.use_dataseglist:
                if datasetting_v.segfile_suffix == 'default':
                    segfilename = 'relPose-gps-gt.csv.seg'
                else:
                    segfilename = 'relPose-gps-gt.csv.seg' + datasetting_v.segfile_suffix
                seglistfile_v = os.path.join(workpath_v, dataset_name_v, branch_v, datasetting_v.algo, segfilename)
            arglist_validating = {}
            arglist_validating['odomfile'] = odomlocal_v
            arglist_validating['odomtradcovfile'] = odomlocaltradcovfile_v
            arglist_validating['laserodomfile'] = laserodomfile_v
            arglist_validating['sceneImgDir'] = imgdir_v
            arglist_validating['frmset'] = (datasetting_v.st_frame, datasetting_v.end_frame)
            arglist_validating['seglistfile'] = seglistfile_v
            arglist_validating['usingseglist'] = datasetting_v.use_dataseglist
            dataset_validating = gen_loc_dataset(time_step_num, gpsgt_v, **arglist_validating)


    elif mode == 'test-acc':
        time_step_num = config.testing.time_step_num
        arglist['odomfile'] = odomlocal
        arglist['odomtradcovfile'] = odomlocaltradcovfile
        arglist['laserodomfile'] = laserodomfile
        arglist['sceneImgDir'] = imgdir
        arglist['frmset'] = (datasetting.st_frame, datasetting.end_frame)
        arglist['lasertradcovfile'] = lasertradcovfile
        arglist['seglistfile'] = seglistfile
        arglist['usingseglist'] = datasetting.use_dataseglist
    elif mode == 'test-pre':
        time_step_num = 1
        arglist['sceneImgDir'] = imgdir
        arglist['frmset'] = (datasetting.st_frame, datasetting.end_frame)
        arglist['usingseglist'] = False
    elif mode == 'test-viewcov':
        time_step_num = 1
        arglist['sceneImgDir'] = imgdir
        arglist['frmset'] = (datasetting.st_frame, datasetting.end_frame)
        arglist['lasertradcovfile'] = lasertradcovfile
        if datasetting.algo == 'LO-Sampling':
            algocomp = 'LO-csmICP'
            lasertradcovfile_comp = os.path.join(workpath, dataset_name, branch, algocomp, 'tradcovmatrix-targetodom.csv')
            arglist['lasertradcovfile-comp'] = lasertradcovfile_comp
        elif datasetting.algo == 'LO-csmICP':
            algocomp = 'LO-Sampling'
            lasertradcovfile_comp = os.path.join(workpath, dataset_name, branch, algocomp, 'tradcovmatrix-targetodom.csv')
            arglist['lasertradcovfile-comp'] = lasertradcovfile_comp
        arglist['usingseglist'] = False

    else:
        raise NotImplementedError

    dataset = gen_loc_dataset(time_step_num, gpsgt, **arglist, **othersection)
    if mode == 'train' and config.training.validating:
        return [dataset, dataset_validating]
    else:
        return [dataset]


def gen_loc_dataset(timesteplen, localGTfile, **kwargs):
    """
    Generation dataset instance accoring to options from kwargs.
    :param timesteplen: positioning iteration.
    :param localGTfile: necessary input, position groud truth file.
    :param kwargs:
    :return: dataset instance
    """
    frmset = kwargs.get("frmset")
    if kwargs.get("allfrm"):
        frmset = None
    seglistfile = None
    if kwargs['usingseglist']:
        seglistfile = kwargs['seglistfile']
    dataset = SeqDataset(timesteplen, frmset=frmset, seglistfile=seglistfile)

    suc = dataset.addSection(localGTfile, 'gtresult')

    if 'odomfile' in kwargs.keys():
        suc &= dataset.addSection(kwargs['odomfile'], 'odomresult')

    if 'laserodomfile' in kwargs.keys():
        suc &= dataset.addSection(kwargs['laserodomfile'], 'loresult')

    if 'odomtradcovfile' in kwargs.keys():
        suc &= dataset.addSection(kwargs['odomtradcovfile'], 'odomtradcov')

    if 'lasertradcovfile' in kwargs.keys():
        suc &= dataset.addSection(kwargs['lasertradcovfile'], 'lotradcov')
        if not suc:
            print("******************No trad cov file found!!!********************")

    if 'lasertradcovfile-comp' in kwargs.keys():
        suc &= dataset.addSection(kwargs['lasertradcovfile-comp'], 'lotradcov-comp')

    if 'vofile' in kwargs.keys():
        suc &= dataset.addSection(kwargs['vofile'], 'voresult')

    if 'votradcovfile' in kwargs.keys():
        suc &= dataset.addSection(kwargs['votradcovfile'], 'votradcov')

    if 'globalgtfile' in kwargs.keys():
        suc &= dataset.addSection(kwargs['globalgtfile'], 'globalgt')

    if 'modelprediction_infomatfile' in kwargs.keys():
        suc &= dataset.addSection(kwargs['modelprediction_infomatfile'], 'modelprediction_infomat')

    if 'sceneImgDir' in kwargs.keys():
        imgbinfile = os.path.join(kwargs['sceneImgDir'], 'img.tbin')
        imgbinInfofile = os.path.join(kwargs['sceneImgDir'], 'imgbininfo.txt')
        if kwargs.get('imgwithtime'):
            imgbinTimefile = os.path.join(kwargs['sceneImgDir'], 'imgbinTime.txt')
            imgloader = ImageTensorBinLoader(imgbinfile, imgbinInfofile, timefile=imgbinTimefile)
        else:
            imgloader = ImageTensorBinLoader(imgbinfile, imgbinInfofile)
        suc &= dataset.registerDataLoader(imgloader, 'sceneImg')

    if kwargs['usingseglist']:
        dataset.check_seg_data()

    return dataset



def stack_data_sample(sample):
    """
    Make all data have tensor form merging from list
    data shape = batchsize*xxx
    :param sample:
    """
    for i in sample.items():
        if i[0] != 'seglength':
            sample[i[0]] = torch.stack(i[1]).float().transpose(0, 1) # force type convert


def change_data_device(data, device):
    """
    Move data to device.
    :param data:
    :param device:
    """
    assert isinstance(data, dict)
    if isinstance(device, torch.device):
        for it in data.items():
            if isinstance(it[1], list):
                for c in it[1]:
                    c = c.to(device)
            elif isinstance(it[1], torch.Tensor):
                data[it[0]] = it[1].to(device)
    elif device == 'cpu':
        for it in data.items():
            if isinstance(it[1], list):
                for c in it[1]:
                    c = c.cpu()
            elif isinstance(it[1], torch.Tensor):
                data[it[0]] = it[1].cpu()
    elif device == 'cuda':
        for it in data.items():
            if isinstance(it[1], list):
                for c in it[1]:
                    c = c.cuda()
            elif isinstance(it[1], torch.Tensor):
                data[it[0]] = it[1].cuda()
    else:
        assert True





