# coding=utf-8
from __future__ import print_function

import sys
import traceback

sys.path.append(".")
from nltk.translate import bleu_score

from utils.dataset import Dataset
from utils.global_names import GlobalNames
from utils.nn_utils import word2id, id2word
from utils.tree_analysis import eval_s2b


class TagJudge(object):
    def __init__(self, vocab):
        self.vocab = vocab
        self.unk = vocab.unk_id

    def is_tag(self, item):
        idx = self.vocab[item]
        return idx > self.unk


def recovery(srcs, vocab):
    return id2word(word2id(srcs, vocab), vocab)


def get_bleu_score(references, hypothesis):
    """
    :type references: list(list(list(str)))
    :param examples: list(examples)
    :param hypothesis: list(list(list(str)))
    :type hypotheses: list(list(str))
    """

    hypothesis = [hyp[0][0] for hyp in hypothesis]

    return 100.0 * bleu_score.corpus_bleu(list_of_references=references, hypotheses=hypothesis)


def get_f1_score(examples, hypothesis):
    golds = [e.tgt for e in examples]
    preds = [hyp[0][0] for hyp in hypothesis]
    cum_acc, error_analysis = eval_s2b(preds, golds)
    print(error_analysis)
    return cum_acc.fscore()


def decode(examples, model):
    was_training = model.training
    model.eval()
    decode_results = model.batch_greedy_decode(examples)
    if was_training: model.train()
    return decode_results


def evaluate(examples, model, eval_src='src', eval_tgt='src', return_decode_result=False, batch_size=None):
    cum_oracle_acc = 0.0
    data_set = Dataset(examples)
    new_examples = []
    decode_results = []
    if batch_size is None:
        batch_size = 5 if "eval_bs" not in model.args else model.args.eval_bs
    for batch_examples in data_set.batch_iter(batch_size, shuffle=False):
        if eval_src == "src":
            batch_examples.sort(key=lambda e: -len(e.src))
        else:
            batch_examples.sort(key=lambda e: -len(e.tgt))
        new_examples.extend(batch_examples)
        _results = decode(batch_examples, model)
        decode_results.extend(_results)
    references = [[recovery(e.src, model.vocab.src)] for e in new_examples] if eval_tgt == "src" else \
        [[recovery(e.tgt, model.vocab.tgt)] for e in new_examples]
    cum_acc = get_bleu_score(references, decode_results)
    eval_result = {'accuracy': cum_acc,
                   'reference': references,
                   'predict': decode_results,
                   'oracle_accuracy': cum_oracle_acc}

    if return_decode_result:
        return eval_result, decode_results
    else:
        return eval_result


def auto_decode(examples, model, args, choice='ae'):
    was_training = model.training
    model.eval()
    if choice == 'ae':
        decode_results = model.greedy_search(examples)
    else:
        decode_results = model.batch_greedy_decode(examples)
    if was_training: model.train()

    return decode_results


def evaluate_autoencoder(examples, auto_encoder, args, choice='ae', return_decode_result=False):
    cum_oracle_acc = 0.0
    data_set = Dataset(examples)
    new_examples = []
    decode_results = []
    for batch_examples in data_set.batch_iter(batch_size=50, shuffle=False):
        batch_examples.sort(key=lambda e: -len(e.src))
        new_examples.extend(batch_examples)
        _results = auto_decode(batch_examples, auto_encoder, args, choice=choice)
        decode_results.extend(_results)
    references = [[recovery(e.src, auto_encoder.vocab)] for e in new_examples]
    cum_acc = get_bleu_score(references, decode_results)
    eval_result = {'accuracy': cum_acc,
                   'oracle_accuracy': cum_oracle_acc}

    if return_decode_result:
        return eval_result, decode_results
    else:
        return eval_result


def evaluate_vae(examples, vae, args, choice='vae', return_decode_result=False):
    cum_oracle_acc = 0.0
    data_set = Dataset(examples)
    new_examples = []
    decode_results = []
    for batch_examples in data_set.batch_iter(batch_size=50, shuffle=False):
        batch_examples.sort(key=lambda e: -len(e.src))
        new_examples.extend(batch_examples)
        _results = auto_decode(batch_examples, vae, args, choice=choice)
        decode_results.extend(_results)
    references = [[recovery(e.src, vae.vocab.src)] for e in new_examples]
    cum_acc = get_bleu_score(references, decode_results)
    eval_result = {'accuracy': cum_acc,
                   'oracle_accuracy': cum_oracle_acc}

    if return_decode_result:
        return eval_result, decode_results
    else:
        return eval_result


def reconstruct(examples, model, args, verbose=False):
    was_training = model.training
    model.eval()

    if model.args.model_select.startswith("MySeq"):
        decode_results = model.batch_greedy_reconstruct(examples)
    else:
        if verbose:
            print('evaluating %d examples' % len(examples))
        decode_results = []
        count = 0
        for example in examples:
            hyps = model.sample(example.src, beam_size=args.beam_size)
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
            decode_results.append(decoded_hyps)
        if verbose:
            print("evaluate finish")

    if was_training: model.train()

    return decode_results


def evaluate_reconstructor(examples, reconstructor, args, verbose=False, return_decode_result=False):
    cum_oracle_acc = 0.0
    if reconstructor.args.model_select.startswith("MySeq"):
        data_set = Dataset(examples)
        new_examples = []
        decode_results = []
        for batch_examples in data_set.batch_iter(batch_size=50, shuffle=False):
            batch_examples.sort(key=lambda e: -len(e.tgt))
            new_examples.extend(batch_examples)
            _results = reconstruct(batch_examples, reconstructor, args, verbose=verbose)
            decode_results.extend(_results)
    else:
        decode_results = reconstruct(examples, reconstructor, args, verbose)
        new_examples = examples

    # references = [[e.src] for e in new_examples]
    references = [[recovery(e.src, reconstructor.vocab.src)] for e in new_examples]
    cum_acc = get_bleu_score(references, decode_results)

    eval_result = {'accuracy': cum_acc,
                   'oracle_accuracy': cum_oracle_acc}

    if return_decode_result:
        return eval_result, decode_results
    else:
        return eval_result


def parse(examples, model, args, verbose=False):
    was_training = model.training
    model.eval()

    if model.args.model_select.startswith("MySeq"):
        decode_results = model.batch_greedy_parse(examples)
    else:
        if verbose:
            print('evaluating %d examples' % len(examples))
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
            decode_results.append(decoded_hyps)
        if verbose:
            print("evaluate finish")

    if was_training: model.train()

    return decode_results


def evaluate_parser(examples, parser, args, verbose=False, return_decode_result=False):
    if args.eval_mode == "F1":
        return eval_parser_f1(examples, parser, args, verbose, return_decode_result)
    else:
        return eval_parser_bleu(examples, parser, args, verbose, return_decode_result)


def eval_parser_bleu(examples, parser, args, verbose=False, return_decode_result=False):
    cum_oracle_acc = 0.0

    if parser.args.model_select.startswith("MySeq"):
        data_set = Dataset(examples)
        new_examples = []
        decode_results = []
        for batch_examples in data_set.batch_iter(batch_size=10, shuffle=False):
            new_examples.extend(batch_examples)
            _results = parse(batch_examples, parser, args, verbose=verbose)
            decode_results.extend(_results)
        references = [[e.tgt] for e in new_examples]
        cum_acc = get_bleu_score(references, decode_results)
    else:
        decode_results = parse(examples, parser, args, verbose)
        references = [[e.tgt] for e in examples]
        cum_acc = get_bleu_score(references, decode_results)

    eval_result = {'accuracy': cum_acc,
                   'oracle_accuracy': cum_oracle_acc}

    if return_decode_result:
        return eval_result, decode_results
    else:
        return eval_result


def eval_parser_f1(examples, parser, args, verbose=False, return_decode_result=False):
    cum_oracle_acc = 0.0
    fm = TagJudge(vocab=parser.vocab.stag)
    GlobalNames.set_judge(fm)
    if parser.args.model_select.startswith("MySeq"):
        data_set = Dataset(examples)
        new_examples = []
        decode_results = []
        for batch_examples in data_set.batch_iter(batch_size=50, shuffle=False):
            new_examples.extend(batch_examples)
            _results = parse(batch_examples, parser, args, verbose=verbose)
            decode_results.extend(_results)

    else:
        decode_results = parse(examples, parser, args, verbose)
        new_examples = examples
    cum_acc = get_f1_score(new_examples, decode_results)

    eval_result = {'accuracy': cum_acc,
                   'oracle_accuracy': cum_oracle_acc}

    if return_decode_result:
        return eval_result, decode_results
    else:
        return eval_result
