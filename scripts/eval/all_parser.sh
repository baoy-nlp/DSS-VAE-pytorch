#!/usr/bin/env bash

pred_output=/home/user_data/baoy/experiments/Semi/model/mh-s/${1}/Test
length_match_pred=${pred_output}/trans.length.txt.unmatch.pred
length_unmatch_pred=${pred_output}/trans.random.txt.unmatch.pred

ref_length_match_ori=${pred_output}/trans.length.txt.unmatch.ori
ref_length_match_tgt=${pred_output}/trans.length.txt.unmatch.tgt
ref_length_unmatch_ori=${pred_output}/trans.random.txt.unmatch.ori
ref_length_unmatch_tgt=${pred_output}/trans.random.txt.unmatch.tgt


cd /home/user_data/baoy/projects/zpar
./dist/zpar.en -oc english-models/ ${length_match_pred} > ${length_match_pred}.par
./dist/zpar.en -oc english-models/ ${length_unmatch_pred} > ${length_unmatch_pred}.par
./dist/zpar.en -oc english-models/ ${ref_length_match_ori} > ${ref_length_match_ori}.par
./dist/zpar.en -oc english-models/ ${ref_length_match_tgt} > ${ref_length_match_tgt}.par
./dist/zpar.en -oc english-models/ ${ref_length_unmatch_ori} > ${ref_length_unmatch_ori}.par
./dist/zpar.en -oc english-models/ ${ref_length_unmatch_tgt} > ${ref_length_unmatch_tgt}.par


pred_output=/home/user_data/baoy/experiments/Semi/model/mh-s/${1}/Test
dev_pred=${pred_output}/dev.para.txt.match.pred
dev_ori=${pred_output}/dev.para.txt.match.ori
dev_tgt=${pred_output}/dev.para.txt.match.tgt

cd /home/user_data/baoy/projects/zpar
./dist/zpar.en -oc english-models/ ${dev_pred} > ${dev_pred}.par
./dist/zpar.en -oc english-models/ ${dev_ori} > ${dev_ori}.par
./dist/zpar.en -oc english-models/ ${dev_tgt} > ${dev_tgt}.par
