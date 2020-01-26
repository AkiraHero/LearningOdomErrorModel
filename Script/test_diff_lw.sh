#!/usr/bin/env bash
DIR='/home/akira/dataprocessing/exp_log_journal/corridor0416/diff_yaw_lw'
file=$(ls ${DIR}|grep epoch79)
for i in $file
do
 echo ${DIR}/$i
  python3.5 ../Main/Test-CompareLocAccuracy.py ../UnitTest/test-csm-s.ini\
  --para_file ${DIR}/$i\
  --trad_cov_scale 0.01\
  --log_dir /home/akira/dataprocessing/exp_log_journal/corridor0416/test-lossweight\
  --device_id 1\
  --time_step_num 50\
  --batch_size 256\
  --algo csm
done