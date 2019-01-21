# coding=utf-8
from __future__ import print_function

import os
import time

from .tools import *


# sys.path.append(".")
# import sys


def decode(examples, model):
    was_training = model.training
    model.eval()
    decode_results = model.predict(examples)
    if was_training: model.train()
    return decode_results


def evaluate(examples, model, eval_src='src', eval_tgt='src', return_decode_result=False, batch_size=None,
             out_dir=None):
    cum_oracle_acc = 0.0
    pred_examples = []

    if batch_size is None:
        batch_size = 50 if "eval_bs" not in model.args else model.args.eval_bs
    inp_examples = prepare_input(
        examples,
        eval_src=eval_src,
        batch_size=batch_size
    )
    ref_examples = []
    eval_start = time.time()
    for batch_examples in inp_examples:
        ref_examples.extend(batch_examples)
        pred_result = decode(batch_examples, model)
        pred_examples.extend(pred_result)
    # references = [[recovery(e.src, model.vocab.src)] for e in inp_examples] if eval_tgt == "src" else \
    #     [[recovery(e.tgt, model.vocab.tgt)] for e in inp_examples]
    use_time = time.time() - eval_start
    references = prepare_ref(ref_examples, eval_tgt, model.vocab)
    acc = get_bleu_score(references, pred_examples)
    eval_result = {'accuracy': acc,
                   'reference': references,
                   'predict': pred_examples,
                   'oracle_accuracy': cum_oracle_acc,
                   'use_time': use_time
                   }

    if out_dir is not None:

        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        pred = predict_to_plain(eval_result['predict'])
        # gold = reference_to_plain(eval_result['reference'])
        gold = con_to_plain(eval_result['reference'])

        write_result(pred, fname=os.path.join(out_dir, "pred.txt"))
        write_result(gold, fname=os.path.join(out_dir, "gold.txt"))

    if return_decode_result:
        return eval_result, pred_examples
    else:
        return eval_result
