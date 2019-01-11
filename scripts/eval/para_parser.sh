#!/usr/bin/env bash

pred_output=/home/user_data/baoy/experiments/Semi/model/mh-s/${1}/Test
dev_pred=${pred_output}/dev.para.txt.match.pred
dev_ori=${pred_output}/dev.para.txt.match.ori
dev_tgt=${pred_output}/dev.para.txt.match.tgt

cd /home/user_data/baoy/projects/zpar
./dist/zpar.en -oc english-models/ ${dev_pred} > ${dev_pred}.par
./dist/zpar.en -oc english-models/ ${dev_ori} > ${dev_ori}.par
./dist/zpar.en -oc english-models/ ${dev_tgt} > ${dev_tgt}.par
