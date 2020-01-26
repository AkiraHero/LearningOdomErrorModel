#!/usr/bin/env bash
parapath=/home/akira/dataprocessing/Train_log/campus0516-LO-Sampling

for epoch in 39 44 49
do
    python3.5 ../Main/Test-ModelPrediction.py ../UnitTest/test-csm-s-campus-0516-1.ini\
    --para_file ${parapath}/$(ls ${parapath}|grep model_params_epoch${epoch}_)\
    --log_dir /home/akira/dataprocessing/Test_log/EpochCovVis\
    --device_id 1\
    --batch_size 256\
    --algo LO-Sampling
done