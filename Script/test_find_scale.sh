#!/usr/bin/env bash
for scale in 0.01 0.1 1 10 100
do
  python3.5 ../Main/Test-CompareLocAccuracy.py ../UnitTest/test-csm-s.ini\
  --para_file /home/akira/dataprocessing/exp_log_journal/corridor0416/1112_final_sampling_BN_LLT_t30_128/model_params_epoch59_step37_cnt2280-20191112162644.pkl\
  --trad_cov_scale ${scale}\
  --log_dir /home/akira/dataprocessing/exp_log_journal/corridor0416/test-scale\
  --device_id 1\
  --time_step_num 50\
  --batch_size 256\
  --algo sampling
done