#!/usr/bin/env bash

pred_output=/home/user_data/baoy/experiments/Semi/model/mh-s/${1}/Test
length_match_pred=${pred_output}/trans.length.txt.unmatch.pred
length_unmatch_pred=${pred_output}/trans.random.txt.unmatch.pred
ref_length_match_ori=${pred_output}/trans.length.txt.unmatch.ori
ref_length_match_tgt=${pred_output}/trans.length.txt.unmatch.tgt
ref_length_unmatch_ori=${pred_output}/trans.random.txt.unmatch.ori
ref_length_unmatch_tgt=${pred_output}/trans.random.txt.unmatch.tgt


cd /home/user_data/baoy/projects/seq2seq_parser
python test/eval_test.py ${length_match_pred} ${ref_length_match_ori}
python test/eval_test.py ${length_match_pred} ${ref_length_match_tgt}
python test/eval_test.py ${length_unmatch_pred} ${ref_length_unmatch_ori}
python test/eval_test.py ${length_unmatch_pred} ${ref_length_unmatch_tgt}
python test/eval_test.py ${ref_length_match_ori} ${ref_length_match_tgt}
python test/eval_test.py ${ref_length_unmatch_ori} ${ref_length_unmatch_tgt}

dev_pred=${pred_output}/dev.para.txt.match.pred
dev_ori=${pred_output}/dev.para.txt.match.ori
dev_tgt=${pred_output}/dev.para.txt.match.tgt

python test/eval_test.py ${dev_pred} ${dev_ori}
python test/eval_test.py ${dev_pred} ${dev_tgt}
python test/eval_test.py ${dev_ori} ${dev_tgt}