#!/usr/bin/env bash

for yaw_lw in `seq 0 20 200`
do
  python3.5 ../Main/TrainLocErrorLearningModel.py ../UnitTest/train-csm-s.ini\
  --yawloss_weight ${yaw_lw}\
  --dir /home/akira/dataprocessing/exp_log_journal/corridor0416/diff_yaw_lw\
  --epoch 80\
  --device_id 0\
  --time_step_num 30\
  --batch_size 256\
  --log_epoch_interval 10000
done