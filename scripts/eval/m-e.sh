#!/usr/bin/env bash
cd ../..
Data_dir=/home/user_data/baoy/experiments/Semi/model/mh-s/${1}/Test
python3 test/eval_test.py ${Data_dir}/${2} ${Data_dir}/${3}
