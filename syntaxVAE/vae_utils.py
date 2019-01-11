import os
import shutil
import sys

import numpy as np
import torch

from syntaxVAE import build_model
from syntaxVAE.vocab import Vocab
from utils.config_utils import dict_to_yaml, args_to_dict
from utils.dataset import Dataset


class Ifilter(object):
    """
    ifilter(function or None, sequence) --> ifilter object

    Return those items of sequence for which function(item) is true.
    If function is None, return the items that are true.
    """

    def next(self):  # real signature unknown; restored from __doc__
        """ x.next() -> the next value, or raise StopIteration """
        pass

    def __getattribute__(self, name):  # real signature unknown; restored from __doc__
        """ x.__getattribute__('name') <==> x.name """
        pass

    def __init__(self, function_or_None, sequence):  # real signature unknown; restored from __doc__
        pass

    def __iter__(self):  # real signature unknown; restored from __doc__
        """ x.__iter__() <==> iter(x) """
        pass

    @staticmethod  # known case of __new__
    def __new__(S, *more):  # real signature unknown; restored from __doc__
        """ T.__new__(S, ...) -> a new object with type S, a subtype of T """
        pass


def log_tracker(ret_loss, tracker):
    for key, val in ret_loss.items():
        if isinstance(val, torch.Tensor):
            if key in tracker:
                tracker[key] = torch.cat((tracker[key], val.data.unsqueeze(0)))
            else:
                tracker[key] = val.data.unsqueeze(0)
    return tracker


def lr_schedule(is_better, model, optimizer, main_args, patience, num_trial,
                model_dir, model_file, epoch, reload_model=True, log_writer=None):
    if is_better:
        patience = 0
        # print('save currently the best model ..', file=sys.stderr)
        print('save model to [%s]' % model_file, file=sys.stderr)
        model.save(model_file)
        # also save the optimizers' state
        torch.save(optimizer.state_dict(), model_file + '.optim.bin')
    elif epoch == main_args.max_epoch:
        print('reached max epoch, stop!', file=sys.stderr)
    elif patience < main_args.patience:
        patience += 1
        print('hit patience %d' % patience, file=sys.stderr)

    if patience == main_args.patience:
        num_trial += 1
        print('hit #%d trial' % num_trial, file=sys.stderr)
        if num_trial == main_args.max_num_trial:
            print('early stop!', file=sys.stderr)

        # decay lr, and restore from previously best checkpoint
        lr = optimizer.param_groups[0]['lr'] * main_args.lr_decay
        print('decay learning rate to %f' % lr, file=sys.stderr)
        # load model
        if reload_model:
            print('load previously best model', file=sys.stderr)
            params = torch.load(model_file, map_location=lambda storage, loc: storage)
            model.load_state_dict(params['state_dict'])
            if main_args.cuda: model = model.cuda()

        # load optimizers
        if main_args.reset_optimizer:
            print('reset optimizer', file=sys.stderr)
            optimizer = torch.optim.Adam(model.inference_model.parameters(), lr=lr)
        else:
            print('restore parameters of the optimizers', file=sys.stderr)
            optimizer.load_state_dict(torch.load(model_file + '.optim.bin'))

        # set new lr
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # reset patience
        patience = 0
    lr = optimizer.param_groups[0]['lr']
    if lr <= 1e-6:
        print('early stop!', file=sys.stderr)

    return model, optimizer, num_trial, patience


def init_parameters(model):
    for param in model.parameters():
        param.data.uniform_(-0.08, 0.08)


def init_optimizer(model, args):
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.995))
    return optimizer


def load_data(main_args):
    train_set = Dataset.from_bin_file(main_args.train_file)
    dev_set = Dataset.from_bin_file(main_args.dev_file)
    return train_set, dev_set


def get_eval_dir(main_args, model_args, mode='Train'):
    if main_args.exp_name is not None:
        dir_path = [model_args.model_file, main_args.exp_name]
    else:
        dir_path = [model_args.model_file]
    model_name = ".".join(dir_path)
    model_dir = os.path.join(main_args.model_dir, model_name)
    model_dir = os.path.join(model_dir, mode)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    return model_dir


def get_exp_info(main_args, model_args, check_dir=True):
    if main_args.exp_name is not None:
        dir_path = [model_args.model_file, main_args.exp_name]
    else:
        dir_path = [model_args.model_file]
    model_name = ".".join(dir_path)
    model_dir = os.path.join(main_args.model_dir, model_name)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    model_dir = os.path.join(model_dir, "model")
    log_file = ".".join([main_args.model_dir.split("/")[-1], model_dir.split("/")[-2]])
    log_dir = os.path.join(main_args.logdir, log_file)

    if check_dir:
        if os.path.exists(log_dir):
            print("{} is not empty, del it?".format(log_dir))
            x = input("Y is del, other is not:")
            if x.lower() == "y":
                shutil.rmtree(log_dir)
                print("rm {}".format(log_dir))
            else:
                raise RuntimeError("target is need redirection")

    if main_args.mode.lower().startswith("train"):
        config_params = {
            'base_configs': args_to_dict(main_args),
            'model_configs': args_to_dict(model_args),
        }

        dict_to_yaml(fname=model_dir + ".config", dicts=config_params)

    print("model dir{}\nlog dir{}".format(
        model_dir,
        log_dir
    ))

    return model_dir, log_dir


def init_model(main_args, model_args, model=None):
    if model is None:
        vocab = Vocab.from_bin_file(main_args.vocab)
        model = build_model(model_args.model_select, args=model_args, vocab=vocab)
        init_parameters(model)
    else:
        vocab = model.vocab
    model.train()
    print(vocab)
    if main_args.cuda:
        # device = torch.device("cuda:0" if torch.cuda.is_available else "cpu")
        # model = torch.nn.DataParallel(model)
        # model.to(device)
        if model_args.model_select == "DVAE":
            model.get_gpu()
        else:
            model.cuda()

    optimizer = init_optimizer(model, main_args)
    # print(model)
    return model, optimizer, vocab


def load_model(main_args, model_args, check_dir=True):
    # model_dir = base_args.model_dir + "." + model_configs.model_file
    model_dir, _ = get_exp_info(main_args, model_args, check_dir)
    model_file = model_dir + '.bin'
    print("...... load model from path:{} ......".format(model_file))
    params = torch.load(model_file, map_location=lambda storage, loc: storage)
    args = params['args']
    vocab = params['vocab']
    model = build_model(model_args.model_select, args=args, vocab=vocab)
    model.load_state_dict(params["state_dict"])
    print(model)
    if main_args.cuda:
        # device = torch.device("cuda:0" if torch.cuda.is_available else "cpu")
        # model = torch.nn.DataParallel(model)
        # model.to(device)
        if model_args.model_select == "DVAE":
            model.get_gpu()
        else:
            model.cuda()
    return model


def eval_ppl(model, dev_set, main_args):
    model.eval()
    cum_loss = 0.
    cum_tgt_words = 0.
    for batch in dev_set.batch_iter(main_args.batch_size):
        loss = -model.score(batch).sum()
        cum_loss += loss.item()
        cum_tgt_words += sum(len(e.src) + 1 for e in batch)  # add ending </s>

    ppl = np.exp(cum_loss / cum_tgt_words)
    model.train()
    return ppl
