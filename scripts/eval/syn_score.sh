#!/usr/bin/env bash

pred_output=/home/user_data/baoy/experiments/Semi/model/mh-s/${1}/Test
length_match_pred=${pred_output}/trans.length.txt.unmatch.pred
length_unmatch_pred=${pred_output}/trans.random.txt.unmatch.pred
ref_length_match_ori=${pred_output}/trans.length.txt.unmatch.ori
ref_length_match_tgt=${pred_output}/trans.length.txt.unmatch.tgt
ref_length_unmatch_ori=${pred_output}/trans.random.txt.unmatch.ori
ref_length_unmatch_tgt=${pred_output}/trans.random.txt.unmatch.tgt

cd /home/user_data/baoy/projects/seq2seq_parser

python data_set/eval_syn.py ${length_match_pred}.par ${ref_length_match_ori}.par
python data_set/eval_syn.py ${length_match_pred}.par ${ref_length_match_tgt}.par
python data_set/eval_syn.py ${length_unmatch_pred}.par ${ref_length_unmatch_ori}.par
python data_set/eval_syn.py ${length_unmatch_pred}.par ${ref_length_unmatch_tgt}.par

python test/eval_test.py ${ref_length_match_ori}.par ${ref_length_match_tgt}.par
python test/eval_test.py ${ref_length_unmatch_ori}.par ${ref_length_unmatch_tgt}.par

dev_pred=${pred_output}/dev.para.txt.match.pred
dev_ori=${pred_output}/dev.para.txt.match.ori
dev_tgt=${pred_output}/dev.para.txt.match.tgt

python test/eval_test.py ${dev_pred}.par ${dev_ori}.par
python test/eval_test.py ${dev_pred}.par ${dev_tgt}.par
python test/eval_test.py ${dev_ori}.par ${dev_tgt}.par