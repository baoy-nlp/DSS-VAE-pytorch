from __future__ import absolute_import

import argparse
import sys
import time

sys.path.append(".")
from syntaxVAE.vae_utils import *
from utils.config_utils import dict_to_args
from utils.config_utils import yaml_load_dict
from tensorboardX import SummaryWriter
from syntaxVAE.evaluation import evaluate
from utils.dataset import to_example
from syntaxVAE.vae_metrics import VaeEvaluator
from syntaxVAE.vae_utils import get_eval_dir


def train_ae(main_args, model_args, model=None):
    train_set, dev_set = load_data(main_args)
    model, optimizer, vocab = init_model(main_args, model_args, model)
    print('begin training, %d training examples, %d dev examples' % (len(train_set), len(dev_set)), file=sys.stderr)
    print('vocab: %s' % repr(vocab.src), file=sys.stderr)
    epoch = 0
    train_iter = 0
    report_loss = report_examples = 0.
    history_dev_scores = []
    num_trial = patience = 0

    model_dir, log_dir = get_exp_info(main_args=main_args, model_args=model_args)
    model_file = model_dir + '.bin'
    writer = SummaryWriter(log_dir)

    while True:
        epoch += 1
        epoch_begin = time.time()
        for batch_examples in train_set.batch_iter(batch_size=main_args.batch_size, shuffle=True):
            train_iter += 1
            optimizer.zero_grad()
            loss = -model.score(batch_examples)
            loss_val = torch.sum(loss).item()
            report_loss += loss_val
            report_examples += len(batch_examples)
            loss = torch.mean(loss)
            loss.backward()

            if main_args.clip_grad > 0.:
                torch.nn.utils.clip_grad_norm_(model.parameters(), main_args.clip_grad)

            optimizer.step()

            if train_iter % main_args.log_every == 0:
                print('\r[Iter %d] encoder loss=%.5f' %
                      (train_iter,
                       report_loss / report_examples),
                      file=sys.stderr, end=" ")

                writer.add_scalar(
                    tag='AutoEncoder/Train/loss',
                    scalar_value=report_loss / report_examples,
                    global_step=train_iter
                )
                writer.add_scalar(
                    tag='optimize/lr',
                    scalar_value=optimizer.param_groups[0]['lr'],
                    global_step=train_iter,
                )

                report_loss = report_examples = 0.

            if train_iter % main_args.dev_every == 0:
                print()
                print('\r[Iter %d] begin validation' % train_iter, file=sys.stderr)
                eval_start = time.time()
                eval_results = evaluate(examples=dev_set.examples, model=model, eval_src='src', eval_tgt='src')
                dev_acc = eval_results['accuracy']
                print('\r[Iter %d] auto_encoder %s=%.5f took %ds' % (
                    train_iter, model.args.eval_mode, dev_acc, time.time() - eval_start),
                      file=sys.stderr)
                writer.add_scalar(
                    tag='AutoEncoder/Dev/%s' % model.args.eval_mode,
                    scalar_value=dev_acc,
                    global_step=train_iter
                )

                is_better = history_dev_scores == [] or dev_acc > max(history_dev_scores)
                history_dev_scores.append(dev_acc)

                writer.add_scalar(
                    tag='AutoEncoder/Dev/best %s' % model.args.eval_mode,
                    scalar_value=max(history_dev_scores),
                    global_step=train_iter
                )

                model, optimizer, num_trial, patience = lr_schedule(
                    is_better=is_better,
                    model_dir=model_dir,
                    model_file=model_file,
                    main_args=main_args,
                    patience=patience,
                    num_trial=num_trial,
                    epoch=epoch,
                    model=model,
                    optimizer=optimizer,
                    reload_model=False
                )

        epoch_time = time.time() - epoch_begin
        print('\r[Epoch %d] epoch elapsed %ds' % (epoch, epoch_time), file=sys.stderr)
        writer.add_scalar(
            tag='AutoEncoder/epoch elapsed',
            scalar_value=epoch_time,
            global_step=epoch
        )


def train_vae(main_args, model_args, model=None):
    ts = time.strftime('%Y-%b-%d-%H:%M:%S', time.gmtime())
    train_set, dev_set = load_data(main_args)
    model, optimizer, vocab = init_model(main_args, model_args, model)

    model_dir, logdir = get_exp_info(main_args=main_args, model_args=model_args)
    model_file = model_dir + '.bin'

    eval_dir = get_eval_dir(main_args=main_args, model_args=model_args, mode='Trains')

    evaluator = VaeEvaluator(
        model=model,
        out_dir=eval_dir,
        train_batch_size=main_args.batch_size,
        eval_batch_size=model_args.eval_bs,

    )

    if model_args.tensorboard_logging:
        writer = SummaryWriter(logdir)
        writer.add_text("model", str(model))
        writer.add_text("args", str(main_args))
        writer.add_text("ts", ts)

    train_iter = main_args.start_iter
    epoch = num_trial = patience = 0
    HISTORY_ELBO = []
    HISTORY_BLEU = []
    max_kl_item = -1
    max_kl_weight = None

    continue_anneal = model_args.peak_anneal

    if model_args.peak_anneal:
        model_args.warm_up = 0

    memory_temp_count = 0

    t_type = torch.Tensor
    adv_select = ["ADVCoupleVAE", "VSAE", "ACVAE", "DVAE", "SVAE"]
    if model_args.model_select in adv_select:
        if not model_args.dis_train:
            x = input("you forget set the dis training?,switch it?[Y/N]")
            model_args.dis_train = (x.lower() == "y")

    adv_training = model_args.dis_train and model_args.model_select in adv_select
    if adv_training:
        print("has the adv training process")
    adv_syn = model_args.adv_syn > 0. or model_args.infer_weight * model_args.inf_sem
    adv_sem = model_args.adv_sem > 0. or model_args.infer_weight * model_args.inf_syn

    print(model_args.dev_item.lower())
    while True:
        epoch += 1
        train_track = {}
        for batch_examples in train_set.batch_iter(batch_size=main_args.batch_size, shuffle=True):
            train_iter += 1
            if adv_training:
                ret_loss = model.get_loss(batch_examples, train_iter, is_dis=True)
                if adv_syn:
                    dis_syn_loss = ret_loss['dis syn']
                    optimizer.zero_grad()
                    dis_syn_loss.backward()
                    if main_args.clip_grad > 0.:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), main_args.clip_grad)
                        # optimizer.step()
                if adv_sem:
                    ret_loss = model.get_loss(batch_examples, train_iter, is_dis=True)
                    dis_sem_loss = ret_loss['dis sem']
                    optimizer.zero_grad()
                    dis_sem_loss.backward()
                    if main_args.clip_grad > 0.:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), main_args.clip_grad)
                        # optimizer.step()

            ret_loss = model.get_loss(batch_examples, train_iter)
            loss = ret_loss['Loss']
            optimizer.zero_grad()
            loss.backward()
            if main_args.clip_grad > 0.:
                torch.nn.utils.clip_grad_norm_(model.parameters(), main_args.clip_grad)

            optimizer.step()
            train_iter += 1
            # tracker = update_track(loss, train_avg_kl, train_avg_nll, tracker)
            train_track = log_tracker(ret_loss, train_track)
            if train_iter % main_args.log_every == 0:
                train_avg_nll = ret_loss['NLL Loss']
                train_avg_kl = ret_loss['KL Loss']
                _kl_weight = ret_loss['KL Weight']
                for key, val in ret_loss.items():
                    writer.add_scalar(
                        'Train-Iter/VAE/{}'.format(key),
                        val.item() if isinstance(val, t_type) else val,
                        train_iter
                    )

                print("\rTrain-Iter %04d, Loss %9.4f, NLL-Loss %9.4f, KL-Loss %9.4f, KL-Weight %6.3f, WD-Drop %6.3f"
                      % (train_iter, loss.item(), train_avg_nll, train_avg_kl, _kl_weight, model.step_unk_rate),
                      end=' ')
                writer.add_scalar(
                    tag='optimize/lr',
                    scalar_value=optimizer.param_groups[0]['lr'],
                    global_step=train_iter,
                )

            if train_iter % main_args.dev_every == 0 and train_iter > model_args.warm_up:
                # dev_track, eval_results = _test_vae(model, dev_set, main_args, train_iter)
                dev_track, eval_results = evaluator.evaluate_reconstruction(examples=dev_set.examples,
                                                                            eval_desc="dev{}".format(train_iter),
                                                                            eval_step=train_iter, write_down=False)
                _weight = model._kl_weight(step=train_iter)
                _kl_item = torch.mean(dev_track['KL Item'])
                # writer.add_scalar("VAE/Valid-Iter/KL Item", _kl_item, train_iter)
                for key, val in dev_track.items():
                    writer.add_scalar(
                        'Valid-Iter/VAE/{}'.format(key),
                        torch.mean(val) if isinstance(val, t_type) else val,
                        train_iter
                    )
                if continue_anneal and model.step_kl_weight is None:
                    if _kl_item > max_kl_item:
                        max_kl_item = _kl_item
                        max_kl_weight = _weight
                    else:
                        if (max_kl_item - _kl_item) > model_args.stop_clip_kl:
                            model.step_kl_weight = max_kl_weight
                            writer.add_text(tag='peak_anneal',
                                            text_string="fixed the kl weight:{} with kl peak:{} at step:{}".format(
                                                max_kl_weight,
                                                max_kl_item,
                                                train_iter
                                            ), global_step=train_iter)
                            continue_anneal = False
                dev_elbo = torch.mean(dev_track['Model Score'])
                writer.add_scalar("Evaluation/VAE/Dev Score", dev_elbo, train_iter)

                # evaluate bleu
                dev_bleu = eval_results['accuracy']
                print()
                print("Valid-Iter %04d, NLL_Loss:%9.4f, KL_Loss: %9.4f, Sum Score:%9.4f BLEU:%9.4f" % (
                    train_iter,
                    torch.mean(dev_track['NLL Loss']),
                    torch.mean(dev_track['KL Loss']),
                    dev_elbo,
                    eval_results['accuracy']), file=sys.stderr
                      )
                writer.add_scalar(
                    tag='Evaluation/VAE/Iter %s' % model.args.eval_mode,
                    scalar_value=dev_bleu,
                    global_step=train_iter
                )
                if model_args.dev_item == "ELBO" or model_args.dev_item.lower() == "para-elbo" or model_args.dev_item.lower() == "gen-elbo":
                    is_better = HISTORY_ELBO == [] or dev_elbo < min(HISTORY_ELBO)
                elif model_args.dev_item == "BLEU" or model_args.dev_item.lower() == "para-bleu" or model_args.dev_item.lower() == "gen-bleu":
                    is_better = HISTORY_BLEU == [] or dev_bleu > max(HISTORY_BLEU)

                HISTORY_ELBO.append(dev_elbo)
                writer.add_scalar("Evaluation/VAE/Best Score", min(HISTORY_ELBO), train_iter)
                HISTORY_BLEU.append(dev_bleu)
                writer.add_scalar("Evaluation/VAE/Best BLEU Score", max(HISTORY_BLEU), train_iter)

                if is_better:
                    writer.add_scalar(
                        tag='Evaluation/VAE/Best %s' % model.args.eval_mode,
                        scalar_value=dev_bleu,
                        global_step=train_iter
                    )
                    writer.add_scalar(
                        tag='Evaluation/VAE/Best NLL-LOSS',
                        scalar_value=torch.mean(dev_track['NLL Loss']),
                        global_step=train_iter
                    )
                    writer.add_scalar(
                        tag='Evaluation/VAE/Best KL-LOSS',
                        scalar_value=torch.mean(dev_track['KL Loss']),
                        global_step=train_iter
                    )
                    if train_iter * 2 > model_args.x0:
                        memory_temp_count = 3

                if model_args.dev_item.lower().startswith("gen") and memory_temp_count > 0:
                    evaluator.evaluate_generation(
                        sample_size=len(dev_set.examples),
                        eval_desc="gen_iter{}".format(train_iter),
                    )
                    memory_temp_count -= 1

                if model_args.dev_item.lower().startswith("para") and memory_temp_count > 0:

                    para_score = evaluator.evaluate_para(
                        eval_dir="/home/user_data/baoy/projects/seq2seq_parser/data/quora-mh/unsupervised",
                        # eval_list=["para.raw.text", "para.text"])
                        #  eval_list=["para.raw.text"])
                        eval_list=["dev.para.txt", "test.para.txt"],
                        eval_desc="para_iter{}".format(train_iter))
                    if memory_temp_count == 3:
                        writer.add_scalar(
                            tag='Evaluation/VAE/Para Dev Ori-BLEU',
                            scalar_value=para_score[0][0],
                            global_step=train_iter
                        )
                        writer.add_scalar(
                            tag='Evaluation/VAE/Para Dev Tgt-BLEU',
                            scalar_value=para_score[0][1],
                            global_step=train_iter
                        )
                        if len(para_score) > 1:
                            writer.add_scalar(
                                tag='Evaluation/VAE/Para Test Ori-BLEU',
                                scalar_value=para_score[1][0],
                                global_step=train_iter
                            )
                            writer.add_scalar(
                                tag='Evaluation/VAE/Para Test Tgt-BLEU',
                                scalar_value=para_score[1][1],
                                global_step=train_iter
                            )
                    memory_temp_count -= 1

                model, optimizer, num_trial, patience = lr_schedule(
                    is_better=is_better,
                    model_dir=model_dir,
                    model_file=model_file,
                    main_args=main_args,
                    patience=patience,
                    num_trial=num_trial,
                    epoch=epoch,
                    model=model,
                    optimizer=optimizer,
                    reload_model=model_args.reload_model,
                )
                model.train()
        elbo = torch.mean(train_track['Model Score'])
        print()
        print("Train-Epoch %02d, Score %9.4f" % (epoch, elbo))
        for key, val in train_track.items():
            writer.add_scalar(
                'Train-Epoch/VAE/{}'.format(key),
                torch.mean(val) if isinstance(val, t_type) else val,
                epoch
            )


def test_vae(main_args, model_args, input_mode=0):
    model = load_model(main_args, model_args, check_dir=False)
    out_dir = get_eval_dir(main_args=main_args, model_args=model_args, mode="Test")
    model.eval()
    if not os.path.exists(out_dir):
        sys.exit(-1)
    if model_args.model_select.startswith("Origin"):
        model_args.eval_bs = 20 if model_args.eval_bs < 20 else model_args.eval_bs
    evaluator = VaeEvaluator(
        model=model,
        out_dir=out_dir,
        eval_batch_size=model_args.eval_bs,
        train_batch_size=main_args.batch_size
    )
    train_exam = Dataset.from_bin_file(args.train_file).examples

    para_eval_dir = "/home/user_data/baoy/projects/seq2seq_parser/data/quora-mh/unsupervised"
    para_eval_list = ["dev.para.txt"]
    # ["dev.para.txt", "test.para.txt"]

    if input_mode == 0:
        print("========dev reconstructor========")
        test_set = Dataset.from_bin_file(args.dev_file)
        evaluator.evaluate_reconstruction(examples=test_set.examples, eval_desc="dev")
        print("finish")
        print("========test reconstructor=======")
        test_set = Dataset.from_bin_file(args.test_file)
        evaluator.evaluate_reconstruction(examples=test_set.examples, eval_desc="test")
        print("finish")
        print("========generating samples=======")
        evaluator.evaluate_generation(corpus_examples=train_exam, sample_size=len(test_set.examples), eval_desc="gen")
        print("finish")
    elif input_mode == 1:
        print("========generating samples=======")
        test_exam = Dataset.from_bin_file(args.test_file).examples
        evaluator.evaluate_generation(corpus_examples=train_exam, sample_size=len(test_exam), eval_desc="gen")
        print("finish")
    elif input_mode == 2:
        print("========generating paraphrase========")
        evaluator.evaluate_para(eval_dir=para_eval_dir, eval_list=para_eval_list)
        print("finish")
    elif input_mode == 3:
        print("========supervised generation========")
        # evaluator.evaluate_control()
        evaluator.evaluate_control(eval_dir=para_eval_dir, eval_list=para_eval_list)
        print("finish")
    elif input_mode == 4:
        trans_eval_list = ["trans.length.txt", "trans.random.txt"]
        print("========style transfer========")
        evaluator.evaluate_style_transfer(eval_dir=para_eval_dir, eval_list=trans_eval_list, eval_desc="unmatch")
        evaluator.evaluate_style_transfer(eval_dir=para_eval_dir, eval_list=para_eval_list, eval_desc="match")
        print("finish")
    elif input_mode == 5:
        print("========random syntax select========")
        evaluator.evaluate_pure_para(eval_dir=para_eval_dir, eval_list=para_eval_list)
        print("finish")
    else:
        raw = input("raw sent: ")
        while not raw.startswith("EXIT"):
            e = to_example(raw)
            words = model.batch_greedy_decode(e)
            print("origin:", " ".join(words[0][0][0]))
            to_ref = input("ref syn : ")
            while not to_ref.startswith("NEXT"):
                syn_ref = to_example(to_ref)
                ret = model.eval_adv(e, syn_ref)
                if not model_args.model_select == "OriginVAE":
                    print("ref syntax: ", " ".join(ret['ref syn'][0][0][0]))
                    print("ori syntax: ", " ".join(ret['ori syn'][0][0][0]))
                print("switch result: ", " ".join(ret['res'][0][0][0]))
                to_ref = input("ref syn: ")
            raw = input("input : ")


def process_args():
    opt_parser = argparse.ArgumentParser()
    opt_parser.add_argument('--config_files', type=str, help='config_files')
    opt_parser.add_argument('--exp_name', type=str, help='config_files')
    opt_parser.add_argument('--load_src_lm', type=str, default=None)
    opt_parser.add_argument('--mode', type=str, default=None)
    opt = opt_parser.parse_args()

    configs = yaml_load_dict(opt.config_files)

    base_args = dict_to_args(configs['base_configs']) if 'base_configs' in configs else None
    baseline_args = dict_to_args(configs['baseline_configs']) if 'baseline_configs' in configs else None
    prior_args = dict_to_args(configs['prior_configs']) if 'prior_configs' in configs else None
    encoder_args = dict_to_args(configs['encoder_configs']) if 'encoder_configs' in configs else None
    decoder_args = dict_to_args(configs['decoder_configs']) if 'decoder_configs' in configs else None
    vae_args = dict_to_args(configs['vae_configs']) if 'vae_configs' in configs else None
    ae_args = dict_to_args(configs["ae_configs"]) if 'ae_configs' in configs else None

    if base_args is not None:
        if opt.mode is not None:
            base_args.mode = opt.mode
        if opt.exp_name is not None:
            base_args.exp_name = opt.exp_name
        if opt.load_src_lm is not None:
            base_args.load_src_lm = opt.load_src_lm

    return {
        'base': base_args,
        "baseline": baseline_args,
        'prior': prior_args,
        'encoder': encoder_args,
        "decoder": decoder_args,
        "vae": vae_args,
        "ae": ae_args,
    }


if __name__ == "__main__":
    config_args = process_args()
    args = config_args['base']
    if args.mode == "train_sent":
        train_vae(args, config_args['vae'])
    elif args.mode == "train_ae":
        train_ae(args, config_args['ae'])
    elif args.mode == "test_vae":
        raw_sent = int(input("select test mode: "))
        test_vae(args, config_args['vae'], input_mode=raw_sent)
    elif args.mode == "test_vaea":
        test_vae(args, config_args['vae'], input_mode=0)
    elif args.mode == "test_generating":
        test_vae(args, config_args['vae'], input_mode=1)
    elif args.mode == "test_paraphrase":
        test_vae(args, config_args['vae'], input_mode=2)
    elif args.mode == "test_control":
        test_vae(args, config_args['vae'], input_mode=3)
    elif args.mode == "test_transfer":
        test_vae(args, config_args['vae'], input_mode=4)
    elif args.mode == "test_pure_para":
        test_vae(args, config_args['vae'], input_mode=5)
    else:
        raise NotImplementedError
