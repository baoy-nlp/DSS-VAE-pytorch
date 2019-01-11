from __future__ import print_function

import sys
import traceback

from utils.global_names import GlobalNames
from utils.tree_analysis import eval_s2b


class TagJudge(object):
    def __init__(self, vocab):
        self.vocab = vocab

    def is_tag(self, item):
        return self.vocab[item] > self.vocab.pad_id


def parse(examples, model, args, verbose=False):
    if verbose:
        print('evaluating %d examples' % len(examples))

    was_training = model.training
    model.eval()

    decode_results = []
    count = 0
    for example in examples:
        hyps = model.predict_syntax(example.src, beam_size=args.beam_size)
        decoded_hyps = []
        for hyp_id, hyp in enumerate(hyps):
            try:
                decoded_hyps.append(hyp)
            except:
                if verbose:
                    print("Exception in converting tree to code:", file=sys.stdout)
                    print('-' * 60, file=sys.stdout)
                    print('example id: %d, hypothesis id: %d' % (example.idx, hyp_id), file=sys.stdout)
                    traceback.print_exc(file=sys.stdout)
                    print('-' * 60, file=sys.stdout)

        count += 1
        if verbose and count % 50 == 0:
            print('\rdecoded %d examples...' % count, file=sys.stdout, end=' ')
        if len(decoded_hyps) >= 1:
            decode_results.append(decoded_hyps[0][0])

    if verbose:
        print("evaluate finish")

    if was_training: model.train()

    return decode_results


def evaluate_parser(examples, parser, args, verbose=False, return_decode_result=False):
    fm = TagJudge(vocab=parser.vocab.stag)
    GlobalNames.set_judge(fm)
    decode_results = parse(examples, parser, args, verbose=verbose)

    golds = [e.tgt for e in examples]
    preds = decode_results
    cum_acc, error_analysis = eval_s2b(preds, golds)
    print(error_analysis)

    eval_result = {'accuracy': cum_acc.fscore(),
                   'oracle_accuracy': cum_acc.fscore()}

    if return_decode_result:
        return eval_result, decode_results
    else:
        return eval_result
